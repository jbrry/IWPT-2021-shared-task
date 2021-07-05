"""
based on the implementation in: https://github.com/Hyperparticle/udify/blob/master/udify/predictors/predictor.py
"""

from typing import Dict, Any, List, Tuple
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

@Predictor.register("conllu-predictor")
class ConlluPredictor(Predictor):
    """
    Predictor that takes in a sentence and returns
    a set of heads and tags for it.
    Predictor for the :class:`~allennlp.models.BiaffineDependencyParser` model
    but extended to write conllu lines.
    """
    def __init__(self,
                model: Model, 
                dataset_reader: DatasetReader,
                ) -> None:
        super().__init__(model, dataset_reader)
    
    def predict(self,
                sentence: str,
                ) -> JsonDict: 
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self,
                        json_dict: JsonDict,
                        ) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"tokens"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = sentence.split()
        tokens = str(tokens)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predict_instance(self,
                        instance: Instance,
                        ) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def dump_line(self,
                outputs: JsonDict
                ) -> str:
        conllu_metadata = outputs["conllu_metadata"]
        word_count = len([word for word in outputs["words"]])

        predicted_arcs = outputs["arcs"]
        predicted_arc_tags = outputs["arc_tags"]
        
        # changes None to "_"
        cleaned_heads = []
        predicted_heads = outputs["head_indices"]
        for head in predicted_heads:
            if type(head) != int:
                head = "_"
                cleaned_heads.append(head)
            else:
                cleaned_heads.append(head)
        outputs["head_indices"] = cleaned_heads
                     

        lines = zip(*[outputs[k] if k in outputs else ["_"] * word_count
                      for k in ["ids", "words", "lemmas", "upos", "xpos", "feats",
                                "head_indices", "head_tags", "arc_tags", "misc"]])

        multiword_map = None
        if outputs["multiword_ids"]:
            multiword_ids = [[id] + [int(x) for x in id.split("-")] for id in outputs["multiword_ids"]]
            multiword_forms = outputs["multiword_forms"]
            multiword_map = {start: (id_, form) for (id_, start, end), form in zip(multiword_ids, multiword_forms)}


        output_lines = []
        for i, line in enumerate(lines):
            line = [str(l) for l in line]

            # Handle multiword tokens
            if multiword_map and i+1 in multiword_map:
                id_, form = multiword_map[i+1]
                row = f"{id_}\t{form}" + "".join(["\t_"] * 8)
                output_lines.append(row)

            row = "\t".join(line)
            output_lines.append(row)

        output_lines = conllu_metadata + output_lines
        return "\n".join(output_lines) + "\n\n"