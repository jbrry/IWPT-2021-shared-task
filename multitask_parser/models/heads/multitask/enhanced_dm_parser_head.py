# source: https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/models/graph_parser.py
# modified by James Barry, Dublin City University
# Licence: Apache License 2.0

"""
This model is based on the original AllenNLP implementation: https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/models/graph_parser.py
"""

from typing import Dict, Tuple, Any, List
import logging
import copy
from operator import itemgetter

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.models.heads.head import Head
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import min_value_of_dtype
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from multitask_parser.training.enhanced_attachment_scores import EnhancedAttachmentScores

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Head.register("enhanced_dm_parser")
class EnhancedDMParser(Head):
    """
    Parsing head to predict enhanced UD graphs.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        encoder_dim: int,
        tag_representation_dim: int,
        arc_representation_dim: int,
        tag_feedforward: FeedForward = None,
        arc_feedforward: FeedForward = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        edge_prediction_threshold: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.edge_prediction_threshold = edge_prediction_threshold
        if not 0 < edge_prediction_threshold < 1:
            raise ConfigurationError(f"edge_prediction_threshold must be between "
                                     f"0 and 1 (exclusive) but found {edge_prediction_threshold}.")


        self.head_arc_feedforward = arc_feedforward or FeedForward(
            encoder_dim, 1, arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        num_labels = self.vocab.get_vocab_size("deps")
        self.head_tag_feedforward = tag_feedforward or FeedForward(
            encoder_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = BilinearMatrixAttention(
            tag_representation_dim, tag_representation_dim, label_dim=num_labels
        )

        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout = Dropout(input_dropout)

        # add a head sentinel to accommodate for extra root token in EUD graphs
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, encoder_dim]))


        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self._enhanced_attachment_scores = EnhancedAttachmentScores()
        self._arc_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._tag_loss = torch.nn.CrossEntropyLoss(reduction="none")
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        encoded_text: TextFieldTensors,
        mask: torch.BoolTensor,
        metadata: List[Dict[str, Any]],
        upos: torch.LongTensor,
        lemmas: torch.LongTensor = None,
        upos_encoded_representation: torch.FloatTensor = None,
        xpos_encoded_representation: torch.FloatTensor = None,
        feats_encoded_representation: torch.FloatTensor = None,
        enhanced_tags: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : TextFieldTensors, required
            The output of `TextField.as_array()`.
        pos_tags : torch.LongTensor, optional (default = None)
            The output of a `SequenceLabelField` containing POS tags.
        metadata : List[Dict[str, Any]], optional (default = None)
            A dictionary of metadata for each batch element which has keys:
                tokens : `List[str]`, required.
                    The original string tokens in the sentence.
        enhanced_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape ``(batch_size, sequence_length, sequence_length)``.
        # Returns
        An output dictionary.
        """

        concatenated_input = [encoded_text]
     
        if upos_encoded_representation is not None:
            concatenated_input.append(upos_encoded_representation)
        if xpos_encoded_representation is not None:
            concatenated_input.append(xpos_encoded_representation)
        if feats_encoded_representation is not None:
            concatenated_input.append(feats_encoded_representation)

        if len(concatenated_input) > 1:
            encoded_text = torch.cat(concatenated_input, -1)

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        encoded_text = self._dropout(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        head_arc_representation = self._dropout(self.head_arc_feedforward(encoded_text))
        child_arc_representation = self._dropout(self.child_arc_feedforward(encoded_text))

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self._dropout(self.head_tag_feedforward(encoded_text))
        child_tag_representation = self._dropout(self.child_tag_feedforward(encoded_text))

        # shape (batch_size, sequence_length, sequence_length)
        arc_scores = self.arc_attention(head_arc_representation, child_arc_representation)

        # shape (batch_size, num_tags, sequence_length, sequence_length)
        arc_tag_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Switch to (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits.permute(0, 2, 3, 1).contiguous()

        # Since we'll be doing some additions, using the min value will cause underflow
        minus_mask = ~mask * min_value_of_dtype(arc_scores.dtype) / 10
        arc_scores = arc_scores + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        arc_probs, arc_tag_probs = self._greedy_decode(arc_scores, arc_tag_logits, mask)

        output_dict = {"arc_probs": arc_probs, "arc_tag_probs": arc_tag_probs, "mask": mask}

        if metadata:
            output_dict["conllu_metadata"] = [meta["conllu_metadata"] for meta in metadata]
            output_dict["ids"] = [meta["ids"] for meta in metadata]
            output_dict["words"] = [meta["words"] for meta in metadata]
            output_dict["lemmas"] = [meta["lemmas"] for meta in metadata]
            output_dict["upos"] = [meta["upos"] for meta in metadata]
            output_dict["xpos"] = [meta["xpos"] for meta in metadata]
            output_dict["feats"] = [meta["feats"] for meta in metadata]
            output_dict["head_tags"] = [meta["head_tags"] for meta in metadata]
            output_dict["head_indices"] = [meta["head_indices"] for meta in metadata]
            output_dict["original_to_new_indices"] = [meta["original_to_new_indices"] for meta in metadata]
            output_dict["misc"] = [meta["misc"] for meta in metadata]
            output_dict["multiword_ids"] = [x["multiword_ids"] for x in metadata if "multiword_ids" in x]
            output_dict["multiword_forms"] = [x["multiword_forms"] for x in metadata if "multiword_forms" in x]

        if enhanced_tags is not None:
            arc_nll, tag_nll = self._construct_loss(
                arc_scores=arc_scores, arc_tag_logits=arc_tag_logits, enhanced_tags=enhanced_tags, mask=mask
            )

            output_dict["loss"] = arc_nll + tag_nll
            output_dict["arc_loss"] = arc_nll
            output_dict["tag_loss"] = tag_nll

            # get human readable output to computed enhanced graph metrics
            output_dict = self.make_output_human_readable(output_dict)

            # predicted arcs, arc_tags
            predicted_arcs = output_dict["arcs"]
            predicted_arc_tags = output_dict["arc_tags"]
            predicted_labeled_arcs = output_dict["labeled_arcs"]

            # gold arcs, arc_tags
            gold_arcs = [meta["arc_indices"] for meta in metadata]
            gold_arc_tags = [meta["arc_tags"] for meta in metadata]
            gold_labeled_arcs = [meta["labeled_arcs"] for meta in metadata]

            tag_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            self._enhanced_attachment_scores(predicted_arcs, predicted_arc_tags, predicted_labeled_arcs, \
                                             gold_arcs, gold_arc_tags, gold_labeled_arcs, tag_mask)

        return output_dict


    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        arc_tag_probs = output_dict["arc_tag_probs"].cpu().detach().numpy()
        arc_probs = output_dict["arc_probs"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        # append arc and label to calculate ELAS
        labeled_arcs = []

        for instance_arc_probs, instance_arc_tag_probs, length in zip(
            arc_probs, arc_tag_probs, lengths
        ):
            arc_matrix = instance_arc_probs > self.edge_prediction_threshold
            edges = []
            edge_tags = []
            edges_and_tags = []
            # dictionary where a word has been assigned a head
            found_heads = {}
            # Note: manually selecting the most probable edge will result in slightly different F1 scores
            # between F1Measure and EnhancedAttachmentScores
            # set each label to False but will be updated as True if the word has a head over the threshold
            for i in range(length):
                found_heads[i] = False

            for i in range(length):
                for j in range(length):
                    if arc_matrix[i, j] == 1:
                        head_modifier_tuple = (i, j)
                        edges.append(head_modifier_tuple)
                        tag = instance_arc_tag_probs[i, j].argmax(-1)
                        edge_tags.append(self.vocab.get_token_from_index(tag, "deps"))
                        # append ((h,m), label) tuple
                        edges_and_tags.append((head_modifier_tuple, self.vocab.get_token_from_index(tag, "deps")))
                        found_heads[j] = True

            # some words won't have found heads so we will find the edge with the highest probability for each unassigned word
            head_information = found_heads.items()
            unassigned_tokens = []
            for (word, has_found_head) in head_information:
                # we're not interested in selecting heads for the dummy ROOT token
                if has_found_head == False and word != 0:
                    unassigned_tokens.append(word)

            if len(unassigned_tokens) >= 1:
                head_choices = {unassigned_token: [] for unassigned_token in unassigned_tokens}

                # keep track of the probabilities of the other words being heads of the unassigned tokens
                for i in range(length):
                    for j in unassigned_tokens:
                        # edge
                        head_modifier_tuple = (i, j)
                        # score
                        probability = instance_arc_probs[i, j]
                        head_choices[j].append((head_modifier_tuple, probability))

                for unassigned_token, edge_score_tuples in head_choices.items():
                    # get the best edge for each unassigned token based on the score which is element [1] in the tuple.
                    best_edge = max(edge_score_tuples, key = itemgetter(1))[0]

                    edges.append(best_edge)
                    tag = instance_arc_tag_probs[best_edge].argmax(-1)
                    edge_tags.append(self.vocab.get_token_from_index(tag, "deps"))
                    edges_and_tags.append((best_edge, self.vocab.get_token_from_index(tag, "deps")))

            arcs.append(edges)
            arc_tags.append(edge_tags)
            labeled_arcs.append(edges_and_tags)

        output_dict["arcs"] = arcs
        output_dict["arc_tags"] = arc_tags
        output_dict["labeled_arcs"] = labeled_arcs
        return output_dict

    def _construct_loss(
        self,
        arc_scores: torch.Tensor,
        arc_tag_logits: torch.Tensor,
        enhanced_tags: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for an adjacency matrix.
        # Parameters
        arc_scores : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate a
            binary classification decision for whether an edge is present between two words.
        arc_tag_logits : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to generate
            a distribution over edge tags for a given edge.
        enhanced_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length).
            The labels for every arc.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.
        # Returns
        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        arc_indices = (enhanced_tags != -1).float()
        # Make the arc tags not have negative values anywhere
        # (by default, no edge is indicated with -1).
        enhanced_tags = enhanced_tags * arc_indices
        arc_nll = self._arc_loss(arc_scores, arc_indices) * mask.unsqueeze(1) * mask.unsqueeze(2)
        # We want the mask for the tags to only include the unmasked words
        # and we only care about the loss with respect to the gold arcs.
        # tag_mask: (batch, sequence_length, sequence_length)
        tag_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * arc_indices
        batch_size, sequence_length, _, num_tags = arc_tag_logits.size()
        original_shape = [batch_size, sequence_length, sequence_length]
        # reshaped_logits: (batch * sequence_length * sequence_length, num_tags)
        reshaped_logits = arc_tag_logits.view(-1, num_tags)
        # reshaped_tags: (batch * sequence_length * sequence_length)
        reshaped_tags = enhanced_tags.view(-1)
        tag_nll = (
            self._tag_loss(reshaped_logits, reshaped_tags.long()).view(original_shape) * tag_mask
        )
        valid_positions = tag_mask.sum()

        arc_nll = arc_nll.sum() / valid_positions.float()
        tag_nll = tag_nll.sum() / valid_positions.float()
        return arc_nll, tag_nll

    @staticmethod
    def _greedy_decode(
        arc_scores: torch.Tensor, arc_tag_logits: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently.
        # Parameters
        arc_scores : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        arc_tag_logits : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length, num_tags) used to
            generate a distribution over tags for each arc.
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length).
        # Returns
        arc_probs : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, sequence_length) representing the
            probability of an arc being present for this edge.
        arc_tag_probs : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, sequence_length, sequence_length)
            representing the distribution over edge tags for a given edge.
        """
        # Mask the diagonal, because we don't self edges.
        inf_diagonal_mask = torch.diag(arc_scores.new(mask.size(1)).fill_(-numpy.inf))
        arc_scores = arc_scores + inf_diagonal_mask
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_logits = arc_tag_logits + inf_diagonal_mask.unsqueeze(0).unsqueeze(-1)
        # Mask padded tokens, because we only want to consider actual word -> word edges.
        minus_mask = ~mask.unsqueeze(2)
        arc_scores.masked_fill_(minus_mask, -numpy.inf)
        arc_tag_logits.masked_fill_(minus_mask.unsqueeze(-1), -numpy.inf)
        # shape (batch_size, sequence_length, sequence_length)
        arc_probs = arc_scores.sigmoid()
        # shape (batch_size, sequence_length, sequence_length, num_tags)
        arc_tag_probs = torch.nn.functional.softmax(arc_tag_logits, dim=-1)
        return arc_probs, arc_tag_probs


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._enhanced_attachment_scores.get_metric(reset)