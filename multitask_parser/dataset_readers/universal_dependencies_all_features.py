# Dataset reader based on UDify:
# https://github.com/Hyperparticle/udify/blob/master/udify/dataset_readers/universal_dependencies.py
# under MIT License.

from typing import Dict, Tuple, List, Any, Callable
import logging

from overrides import overrides
from conllu import parse_incr

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

logger = logging.getLogger(__name__)


def process_multiword_and_elided_tokens(annotation):
    """
    Processes CoNLL-U ids for multi-word tokens and elided tokens.
    When a token is a MWT, the id is set to None so that the token is not 
    used by the model. Elided token ids are returned as tuples by the conllu library 
    and are converted to a number id here.
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
@DatasetReader.register("universal_dependencies_all_features", exist_ok=True)
class UniversalDependenciesAllFeaturesDatasetReader(DatasetReader):
    """
    Reads a file in the CoNLLU Universal Dependencies format.
    Produces an iterable over all CoNLL-U features, e.g.:
    `id`, `form`, `lemma`, `upos`, `xpos`, `morph. feats`, `heads`, `dep labels`, `deps` and `misc` column.
    # Parameters
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    tokenizer : `Tokenizer`, optional (default = `None`)
        A tokenizer to use to split the text. This is useful when the tokens that you pass
        into the model need to have some particular attribute. Typically it is not necessary.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.tokenizer = tokenizer

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                annotation = process_multiword_and_elided_tokens(annotation)
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
                # considers all tokens except MWTs and elided tokens for prediction
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
                
                heads = get_field("head")
                dep_rels = get_field("deprel")
                deps = get_field("deps")
                dependencies = list(zip(dep_rels, heads))

                misc = get_field("misc", lambda x: "|".join(k + "=" + v if v is not None else k + "=" + "" for k, v in x.items())
                                    if hasattr(x, "items") else "_")

                yield self.text_to_instance(words, lemmas, upos_tags, xpos_tags,
                                            feats, dependencies, ids,
                                            multiword_ids, multiword_forms)

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
    ) -> Instance:

        """
        # Parameters
        words : `List[str]`, required.
            The words in the sentence to be encoded.
        dependencies : `List[Tuple[str, int]]`, optional (default = `None`)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.
        # Returns
        An instance containing words, upos tags, dependency head tags and head
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

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies], text_field, label_namespace="head_index_tags"
            )

        fields["metadata"] = MetadataField({"words": words, "upos": upos_tags, "xpos": xpos_tags})

        fields["metadata"] = MetadataField({
            "words": words,
            "upos": upos_tags,
            "xpos": xpos_tags,
            "feats": feats,
            "lemmas": lemmas,
            "ids": ids,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms
        })

        return Instance(fields)