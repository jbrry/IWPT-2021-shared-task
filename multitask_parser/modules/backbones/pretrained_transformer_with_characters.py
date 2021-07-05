import pickle
from typing import Dict, Optional

from overrides import overrides
import torch
from torch.nn.modules import Dropout

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout


from allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from allennlp.nn import util


@Backbone.register("pretrained_transformer_with_characters")
class PretrainedTransformerWithCharactersBackbone(Backbone):
    """
    Registered as a `Backbone` with name "pretrained_transformer_with_characters".
    A TextFieldEmbedder is used to encode the word representations, where the
    TextFieldEmbedder should be a pretrained_transformer*. In this way, we do not have a second encoder
    for the word features. There are, however, encoders to encode the character representations which are
    then concatenated with the output of a pretrained transformer.
    # Parameters
    vocab : `Vocabulary`
        Necessary for converting input ids to strings in `make_output_human_readable`.  If you set
        `output_token_strings` to `False`, or if you never call `make_output_human_readable`, then
        this will not be used and can be safely set to `None`.
    token_embedders : Dict[str, TextFieldEmbedder]
        A dictionary mapping a key of a dataset name to the appropriate TextFieldEmbedder. In this way,
        a dataset will have its own embeddings. NOTE: in order to do this, in your dataset reader, please make
        sure each dataset has its own token_indexer so that it will have different indexing and thus embeddings.
    encoder: Seq2SeqEncoder
        The encoder which encodes some embedded text. At the moment, there is just a single encoder but it is
        possible that we will introduce multiple encoders (one for each dataset).
    output_token_strings : `bool`, optional (default = `True`)
        If `True`, we will add the input token ids to the output dictionary in `forward` (with key
        "token_ids"), and convert them to strings in `make_output_human_readable` (with key
        "tokens").  This is necessary for certain demo functionality, and it adds only a trivial
        amount of computation if you are not using a demo.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            *,
            word_embedder: TextFieldEmbedder = None,
            token_character_embedder: TextFieldEmbedder = None,
            sentence_character_embedder: TextFieldEmbedder = None,
            word_encoder: Seq2SeqEncoder = None,
            token_character_encoder: Seq2VecEncoder = None,
            sentence_character_encoder: Seq2SeqEncoder = None,            
            output_token_strings: bool = True,
            dropout: float = 0.0,
            input_dropout_word: float = 0.0,
            input_dropout_character: float = 0.0,
            pretrained_model_name: str = None,
    ) -> None:
        super().__init__()
        self._vocab = vocab
        character_vocab = self._vocab.get_token_to_index_vocabulary("sentence_character_vocab")

        self._space_index = character_vocab.get(" ")
        self._padding_index = 0  # NOTE: some indexers pad things as -1

        self._word_embedder = word_embedder

        self._token_character_embedder = token_character_embedder
        self._sentence_character_embedder = sentence_character_embedder

        self._word_encoder = word_encoder
        self._token_character_encoder = token_character_encoder
        self._sentence_character_encoder = sentence_character_encoder

        if self._sentence_character_encoder:
            # add a dummy_tensor which we will use for padding, after we have collected the start and end indices of the words.
            self._dummy_tensor = torch.nn.Parameter(
                torch.zeros([self._sentence_character_encoder.get_output_dim() * 2]))

        self._output_token_strings = output_token_strings
        self._dropout = InputVariationalDropout(dropout)
        self._input_dropout_word = Dropout(input_dropout_word)
        self._input_dropout_character = Dropout(input_dropout_character)

    def find_word_start_and_end_indices(self, character_tensor):
        """
        Iterate over the characters of each sentence to find
        the locations of the start and end characters of each word.
        The start and end locations of each word are then assigned to a tuple.
        character_tensor : TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]
        """

        padding_index = 0
        start_locations = []
        end_locations = []

        # Access the tensor object which is wrapped by two dictionaries.
        character_tensor = character_tensor["token_characters"]["tokens"]

        for sentence in character_tensor:
            start_of_sentence = True
            have_seen_space_char = False
            sentence_starts = []
            sentence_ends = []
            for i, char in enumerate(sentence):
                if start_of_sentence or last_was_space_char:
                    sentence_starts.append(i)
                    start_of_sentence = False
                    last_was_space_char = False
                else:
                    if char == self._space_index:
                        end_of_word_idx = i - 1
                        sentence_ends.append(end_of_word_idx)
                        last_was_space_char = True
                        have_seen_space_char = True
                    elif char == self._padding_index:
                        end_of_word_idx = i - 1
                        sentence_ends.append(end_of_word_idx)
                        break

            # Sentence may not contain any space characters: append last character of word.
            if not have_seen_space_char:
                sentence_ends.append(i)

            # Some words are unfinished and we can't always rely on padding to infer the end of the sentence: append last character in sentence.
            if len(sentence_starts) > len(sentence_ends):
                assert i == (len(sentence) - 1)
                sentence_ends.append(i)

            if len(sentence_starts) >= 1 and len(sentence_ends) >= 1:
                start_locations.append(sentence_starts)
                end_locations.append(sentence_ends)
            else:
                raise ValueError("Something has gone wrong with the indexing, no indices were found.")

        # Gather the start and end indices into start and end tuples.
        location_tuples = []
        for sentence_starts, sentence_ends in zip(start_locations, end_locations):
            current_sent = []
            for word_start, word_end in zip(sentence_starts, sentence_ends):
                word_start_and_end = (word_start, word_end)
                current_sent.append(word_start_and_end)
            location_tuples.append(current_sent)

        assert len(location_tuples) == character_tensor.size(0), \
            print(
                f"collected {len(location_tuples)} location tuples, for a tensor with {character_tensor.size(0)} elements")

        return location_tuples

    def concatenate_word_start_and_end_indices(self, encoded_characters, word_starts_and_ends):
        """
        Takes in the encoded text of characters and concatenates the specific locations of the start and end of each word.
        encoded_characters: `torch.Tensor`
            A tensor of shape (batch_size, sequence_len, char_hidden_size * 2)
        word_starts_and_ends: `List[List[Tuple[int, int]]]`,
            Contains a list of tuples for each element in the batch, where each tuple contains the
            locations of the first and last character of a word in the sentence, e.g.:
                [(0, 4), (6, 8), (10, 10), (12, 13), (15, 21), (23, 23)]
        """
        # List to store the concatenated first and last characters of each word for each sentence in the batch.
        concatenated_tensor_list = []
        for i, start_end_tuple in enumerate(word_starts_and_ends):
            # store representations for the specific sentence.
            sentence_tensor_list = []
            for start, end in start_end_tuple:
                # word_representation is the first and last character representation of a word concatenated together.
                word_representation = torch.cat((encoded_characters[i][start], encoded_characters[i][end]), 0)
                sentence_tensor_list.append(word_representation)
            concatenated_tensor_list.append(sentence_tensor_list)

        return concatenated_tensor_list

    def pad_character_representations(self, tensors):
        """Because we extract the start and end indices from already-encoded text (which is padded),
        we need to add padding to our condensed representation which are just the start and end locations concatenated together."""

        lengths = [len(tensor) for tensor in tensors]
        max_len = max(lengths)

        padded = []
        for sentence in tensors:
            padding = []
            if len(sentence) < max_len:
                diff = max_len - len(sentence)
                for i in range(diff):
                    padding.append(self._dummy_tensor)
                padded_sentence = sentence + padding
                padded.append(padded_sentence)
            else:
                # No padding required for sentences which are of length max_len
                padded.append(sentence)

        return padded

    def convert_to_stacked_tensor(self, padded):
        """Converts a list-of-lists of tensors to a single stacked tensor."""

        stack = []
        for padded_tensor_list in padded:
            # converts a list to a tensor object
            tensor = torch.stack(padded_tensor_list)
            # append tensor to stack
            stack.append(tensor)

        # stack all tensors
        stacked = torch.stack(stack)
        encoded_first_and_last_characters = stacked

        return encoded_first_and_last_characters

    def forward(self,
                words: TextFieldTensors,
                token_characters: TextFieldTensors = None,
                sentence_characters: TextFieldTensors = None,
                head_tags: torch.LongTensor = None,
                head_indices: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:  # type: ignore
        """
        words: tensor of words.
        token_characters: tensor of characters which comprise the words of the sentence.
        sentence_characters: tensor of characters spanning the whole sentence.
        Returns:
            encoded_text: The above features will be encoded and concatenated together.
        """

        if words and len(words) != 1:
            raise ValueError(
                "WordCharBackbone is only compatible with using single TokenIndexers"
            )

        if token_characters and len(token_characters) != 1:
            raise ValueError(
                "WordCharBackbone is only compatible with using single TokenIndexers"
            )

        if sentence_characters and len(sentence_characters) != 1:
            raise ValueError(
                "WordCharBackbone is only compatible with using single TokenIndexers"
            )

        #print(words["tokens"]["token_ids"])

        word_mask = util.get_text_field_mask(words)

        # Word view
        # pretrained transformer just requires an embedder
        if self._word_embedder:
            embedded_words = self._word_embedder(words)
            embedded_words = self._input_dropout_word(embedded_words)
            # encoder if using a seq2seq or seq2vec model
            if self._word_encoder:
                encoded_words = self._word_encoder(embedded_words, word_mask)
                encoded_words = self._dropout(encoded_words)
            else:
                encoded_words = embedded_words
        else:
            encoded_words = None

        # Token-level character view, see character_encoding
        if token_characters:
            token_character_mask = util.get_text_field_mask(token_characters, num_wrapping_dims=1)

            batch_size, sequence_length, word_length = token_character_mask.size()

            embedded_token_characters = self._token_character_embedder(token_characters)

            # Shape: (batch_size * sequence_length, max_characters, embedding_dim)
            embedded_token_characters = embedded_token_characters.view(batch_size * sequence_length, word_length, -1)
            embedded_token_characters = self._input_dropout_character(embedded_token_characters)

            # Shape: (batch_size * sequence_length, max_characters)
            token_character_mask = token_character_mask.view(batch_size * sequence_length, word_length)

            # Shape: (batch_size * sequence_length, max_characters, encoding_dim * 2)
            encoded_token_characters = self._token_character_encoder(embedded_token_characters, token_character_mask)

            # Shape: (batch_size, sequence_length, encoding_dim * 2)
            encoded_token_characters = encoded_token_characters.view(batch_size, sequence_length, -1)
            encoded_token_characters = self._dropout(encoded_token_characters)
        else:
            encoded_token_characters = None

        # Sentence-level character view
        if sentence_characters:
            sentence_character_mask = util.get_text_field_mask(sentence_characters)

            embedded_sentence_characters = self._sentence_character_embedder(sentence_characters)
            embedded_sentence_characters = self._input_dropout_character(embedded_sentence_characters)
            encoded_sentence_characters = self._sentence_character_encoder(embedded_sentence_characters,
                                                                           sentence_character_mask)
            encoded_sentence_characters = self._dropout(encoded_sentence_characters)

            word_starts_and_ends = self.find_word_start_and_end_indices(sentence_characters)
            concatenated_word_start_and_end_representations = self.concatenate_word_start_and_end_indices(
                encoded_sentence_characters, word_starts_and_ends)
            padded_tensor_list = self.pad_character_representations(concatenated_word_start_and_end_representations)
            encoded_sentence_characters = self.convert_to_stacked_tensor(padded_tensor_list)
        else:
            encoded_sentence_characters = None

        # We have condensed the character representations to the same dimensionality as words, so we just need the word_mask from now on.
        mask = word_mask

        # TODO: This all assumes you are encoding words
        if torch.is_tensor(encoded_words):
            encoded_text = encoded_words
        if torch.is_tensor(encoded_token_characters):
            encoded_text = torch.cat([encoded_text, encoded_token_characters], -1)
        if torch.is_tensor(encoded_sentence_characters):
            encoded_text = torch.cat([encoded_text, encoded_sentence_characters], -1)

        outputs = {"encoded_text": encoded_text, "mask": mask}

        return outputs

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not self._output_token_strings:
            return output_dict

        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self._vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict