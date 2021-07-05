from typing import Dict, Optional, List, Any, cast

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder, InputVariationalDropout
from allennlp.models.heads.head import Head
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
import allennlp.nn.util as util
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure


@Head.register("multitask_tagger")
class MultiTaskTaggerHead(Head):
    """
    A Tagging component as part of a MultiTask model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        encoder_dim: int,
        task: str = None,
        #feedforwards: Dict[str, FeedForward] = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.task = task
        self.num_tags = self.vocab.get_vocab_size(self.task)
        self.dropout = InputVariationalDropout(dropout)
        self._feedforward = feedforward

        # Apply a FeedForward network to the output of the encoder
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = encoder_dim

        # Classification layer
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))

        # Attention between UPOS and XPOS # TODO
        # self.tag_attention = BilinearMatrixAttention(
        #     upos_representation_dim, xpos_representation_dim, use_input_biases=True
        # )

        labels = self.vocab.get_index_to_token_vocabulary(self.task)

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

        if feedforward is not None:
            check_dimensions_match(
                encoder_dim,
                feedforward.get_input_dim(),
                "encoder output dim",
                "feedforward input dim",
            )

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        encoded_text: TextFieldTensors,
        mask: torch.BoolTensor,
        lemmas: torch.LongTensor = None,
        upos: torch.LongTensor = None,
        xpos: torch.LongTensor = None,
        feats: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:

        # TODO: when tagging XPOS, concatenate the representation from UPOS. Or do bilinear attention

        if self._feedforward is not None:
            encoded_representation = self._feedforward(encoded_text)
        else:
            encoded_representation = encoded_text

        batch_size, sequence_length, _ = encoded_representation.size()

        #attended_tags = self.tag_attention(upos_, child_arc_representation)

        logits = self.tag_projection_layer(encoded_representation)
        reshaped_log_probs = logits.view(-1, self.num_tags)

        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                        sequence_length,
                                                                        self.num_tags])

        output_dict = {
            "logits": logits, 
            "class_probabilities": class_probabilities, 
            "encoded_representation": encoded_representation
            }

        if self.task == "upos":
            tags = upos
        elif self.task == "xpos":
            tags = xpos
        elif self.task == "feats":
            tags = feats

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits, tags, mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()    
        predictions_list = [all_predictions]
        
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self.task)
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict[f'{self.task}'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

        return metrics_to_return

    default_predictor = "sentence_tagger"