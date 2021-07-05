# sources:
#  * https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/universal_dependencies.py
#    under Apache License 2.0
#  * https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/universal_dependencies.py
#    under MIT License
# modified by James Barry, Dublin City University
# Licence: Apache License 2.0
# Justification of licence choice:
#  * https://www.quora.com/Is-the-MIT-license-compatible-with-the-Apache-License-Version-2-APLv2
#  * https://opensource.stackexchange.com/questions/1711/combining-code-written-under-different-licenses-eiffel-forum-license-mit-and-a


"""
based on the `universal_dependencies` dataset reader in: https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/universal_dependencies.py
and the implementation in: https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/universal_dependencies.py
"""
from typing import Dict, Tuple, List, Any, Callable
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ListField, TextField, SequenceLabelField, MetadataField
from multitask_parser.fields.rooted_adjacency_field import RootedAdjacencyField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


def process_multiword_and_elided_tokens(annotation):
    """
    Processes CoNLL-U ids for multi-word tokens and elided tokens.
    When a token is a MWT, the id is set to None so the token is not used in the model.
    Elided token ids are returned as tuples by the conllu library and are converted to a number id here.
    """

    for i in range(len(annotation)):
        conllu_id = annotation[i]["id"]
        if type(conllu_id) == tuple:
            if "-" in conllu_id:
                conllu_id = str(conllu_id[0]) + "-" + str(conllu_id[2])
                annotation[i]["multi_id"] = conllu_id
                annotation[i]["id"] = None
                annotation[i]["elided_id"] =  None
            elif "." in conllu_id:
                conllu_id = str(conllu_id[0]) + "." + str(conllu_id[2])
                conllu_id = float(conllu_id)
                annotation[i]["elided_id"] = conllu_id
                annotation[i]["id"] = conllu_id
                annotation[i]["multi_id"] = None
        else:
            annotation[i]["elided_id"] =  None
            annotation[i]["multi_id"] = None

    return annotation

# exist_ok has to be true until we remove this from the core library
@DatasetReader.register("universal_dependencies_enhanced", exist_ok=True)
class UniversalDependenciesEnhancedDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.
    # Parameters
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the tokens TextField.
    tokenizer : ``Tokenizer``, optional, default = None
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        read_predicted_from_misc: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenizer = tokenizer
        self.read_predicted_from_misc = read_predicted_from_misc

    def _convert_deps_to_nested_sequences(self, deps):
        """
        Converts a series of deps labels into relation-lists and head-lists respectively.
        # Parameters
        deps : ``List[List[Tuple[str, int]]]``
            The enhanced dependency relations.
        # Returns
        List-of-lists containing the enhanced tags and heads.
        """
        rels = []
        heads = []

        for target_output in deps:
            # check if there is just 1 head
            if len(target_output) == 1:
                rel = [x[0] for x in target_output]
                head = [x[1] for x in target_output]
                rels.append(rel)
                heads.append(head)
            # more than 1 head
            else:
                # append multiple current target heads/rels together respectively
                current_rels = []
                current_heads = []
                for rel_head_tuple in target_output:
                    current_rels.append(rel_head_tuple[0])
                    current_heads.append(rel_head_tuple[1])
                heads.append(current_heads)
                rels.append(current_rels)

        return rels, heads


    def _process_elided_tokens(self, ids, heads):
        """
        Changes elided token format from tuples to float values.
        We create a dictionary which maps the original CoNLL-U indices to
        indices based on the order they appear in the sentence (offsets).
        This means that when an elided token is encountered, e.g. "8.1",
        we map the index to "9" and offset every other index following this token by +1.
        This process is done every time an elided token is encountered.
        At decoding the time, the (head:dependent) tuples are converted back to the original indices.
        # Parameters
        ids : ``List[Union[int, tuple]``
            The original CoNLLU indices of the tokens in the sentence. They will be
            used as keys in a dictionary to map from original to new indices.
        """
        processed_heads = []
        # store the indices of words as they appear in the sentence
        original_to_new_indices = {}
        # set a placeholder for ROOT
        original_to_new_indices[0] = 0

        for token_index, head_list in enumerate(heads):
            conllu_id = ids[token_index]
            # map the original CoNLL-U IDs to the new 1-indexed IDs
            original_to_new_indices[conllu_id] = token_index + 1
            current_heads = []
            for head in head_list:
                # convert copy node tuples: (8, '.', 1) to float: 8.1
                if type(head) == tuple:
                    # join the values in the tuple
                    copy_node = str(head[0]) + '.' + str(head[2])
                    copy_node = float(copy_node)
                    current_heads.append(copy_node)
                else:
                    # regular head id
                    current_heads.append(head)
            processed_heads.append(current_heads)

        # change the indices of the heads to reflect the new order
        offset_heads = []
        for head_list in processed_heads:
            current_heads = []
            for head in head_list:
                if head in original_to_new_indices.keys():
                    # take the 1-indexed head based on the order of words in the sentence
                    offset_head = original_to_new_indices[head]
                    current_heads.append(offset_head)
            offset_heads.append(current_heads)

        return original_to_new_indices, offset_heads


    def _copy_enhanced_dependencies_for_elided_tokens(self, head_indices, head_tags, enhanced_deps):
        """In the basic tree, elided tokens have an empty head and dependency
        label "_". Here, we search for the token and then take that token's head in the enhanced graph"""

        for i, head in enumerate(head_indices):
            # conllu indexing
            modifier = i + 1
            if head == "_":
                # scan through enhanced deps and take the edge from the edeps
                for edep in enhanced_deps:
                    h_m_edge = edep[0]
                    label = edep[1]
                    enhanced_modifier = h_m_edge[1]

                    if modifier == enhanced_modifier:
                        enhanced_head = h_m_edge[0]
                        head_indices[i] = enhanced_head
                        head_tags[i] = label.split(":")[0] # remove lemma TODO: might break for certain dep labels

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                conllu_metadata = []
                metadata = annotation.metadata
                for k, v in metadata.items():
                    metadata_line = (f"# {k} = {v}")
                    conllu_metadata.append(metadata_line)

                self.contains_elided_token = False
                annotation = process_multiword_and_elided_tokens(annotation)
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
                elided_tokens = [x for x in annotation if x["elided_id"] is not None]
                if len(elided_tokens) >= 1:
                    self.contains_elided_token = True

                # considers all tokens except MWTs for prediction
                annotation = [x for x in annotation if x["id"] is not None]

                if len(annotation) == 0:
                    continue

                def get_field(
                            tag: str,
                            map_fn: Callable[[Any], Any] = None,
                            ) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]

                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]

                words = get_field("form")
                lemmas = get_field("lemma")
                upos_tags = get_field("upos")
                xpos_tags = get_field("xpos")
                feats = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                                     if hasattr(x, "items") else "_")

                misc = get_field("misc", lambda x: "|".join(k + "=" + v if v is not None else k + "=" + "" for k, v in x.items())
                                    if hasattr(x, "items") else "_")

                heads = get_field("head")
                dep_rels = get_field("deprel")
                dependencies = list(zip(dep_rels, heads))
                deps = get_field("deps")

                yield self.text_to_instance(words, lemmas, upos_tags, xpos_tags,
                                            feats, dependencies, deps, ids, misc,
                                            multiword_ids, multiword_forms, conllu_metadata)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        feats: List[str] = None,
        dependencies: List[Tuple[str, int]] = None,
        deps: List[List[Tuple[str, int]]] = None,
        ids: List[str] = None,
        misc: List[str] = None,
        multiword_ids: List[str] = None,
        multiword_forms: List[str] = None,
        conllu_metadata: List[str] = None,
        contains_elided_token: bool = False,
    ) -> Instance:

        """
        # Parameters
        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies : ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.
        deps : ``List[List[Tuple[str, int]]]``, optional (default = None)
            A list of lists of (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.
        # Returns
        An instance containing tokens, pos tags, basic and enhanced dependency head tags and head
        indices as fields.
        """

        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]
        
        text_field = TextField(tokens, self._token_indexers)
        fields["words"] = text_field

        names = ["upos", "xpos", "feats", "lemmas"]
        all_tags = [upos_tags, xpos_tags, feats, lemmas]

        for name, field in zip(names, all_tags):
            if field:
                fields[name] = SequenceLabelField(field, text_field, label_namespace=name)

        # Enhanced dependencies
        if deps is not None:
            enhanced_arc_tags, enhanced_arc_indices = self._convert_deps_to_nested_sequences(deps)
            # extra processing is needed if a sentence contains an elided token
            if self.contains_elided_token:
                original_to_new_indices, offset_heads = self._process_elided_tokens(ids, enhanced_arc_indices)
                enhanced_arc_indices = offset_heads
            else:
                original_to_new_indices = None

            assert len(enhanced_arc_tags) == len(enhanced_arc_indices), "each arc should have a label"

            # Prepare labeled edges for Adjacency Matrix
            arc_indices = []
            arc_tags = []
            arc_indices_and_tags = []

            for modifier, head_list in enumerate(enhanced_arc_indices, start=1):
                for head in head_list:
                    arc_indices.append((head, modifier))

            for relation_list in enhanced_arc_tags:
                for relation in relation_list:
                    arc_tags.append(relation)

            assert len(arc_indices) == len(arc_tags), "each arc should have a label"

            for arc_index, arc_tag in zip(arc_indices, arc_tags):
                arc_indices_and_tags.append((arc_index, arc_tag))

            if arc_indices is not None and arc_tags is not None:
                token_field_with_root = ['root'] + tokens
                fields["enhanced_tags"] = RootedAdjacencyField(arc_indices, token_field_with_root, arc_tags, label_namespace="deps")

        # Basic dependency tree
        if dependencies is not None:
            head_tags = [x[0] for x in dependencies]
            head_indices = [x[1] for x in dependencies]

            # Elided words don't have head or label information in the basic tree,
            # so we will copy the head from the enhanced graph.
            if "_" in head_indices:
                self._copy_enhanced_dependencies_for_elided_tokens(head_indices, head_tags, arc_indices_and_tags)

            fields["head_tags"] = SequenceLabelField(
                head_tags, text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                head_indices, text_field, label_namespace="head_index_tags"
            )

        # Now get predicted trees from misc column (if they are available)
        if misc is not None and self.read_predicted_from_misc:
            predicted_head_tags = []
            predicted_head_indices = []

            for predicted_h_m in misc:
                predicted_head = predicted_h_m.split("|")[0]
                predicted_label = predicted_h_m.split("|")[1]

                predicted_head = int(predicted_head.split("=")[1])
                predicted_label = predicted_label.split("=")[1]

                predicted_head_tags.append(predicted_label)
                predicted_head_indices.append(predicted_head)
            
            fields["predicted_head_tags"] = SequenceLabelField(
                predicted_head_tags, text_field, label_namespace="predicted_head_tags"
            )
            fields["predicted_head_indices"] = SequenceLabelField(
                predicted_head_indices, text_field, label_namespace="predicted_head_index_tags"
            )

        fields["metadata"] = MetadataField({
            "words": words,
            "upos": upos_tags,
            "xpos": xpos_tags,
            "feats": feats,
            "lemmas": lemmas,
            "ids": ids,
            "misc": misc,
            "original_to_new_indices": original_to_new_indices,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "predicted_head_tags": predicted_head_tags,
            "predicted_head_indices": predicted_head_indices,
            "arc_indices": arc_indices,
            "arc_tags": arc_tags,
            "labeled_arcs": arc_indices_and_tags,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms,
            "conllu_metadata": conllu_metadata
        })

        return Instance(fields)
