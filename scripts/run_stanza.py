# source: https://github.com/coastalcph/koepsala-parser/blob/60258a81a2db91ca012be4486ea8475376756374/bash/get_preprocessed/get_preprocessed_dev_stanza.py

import logging
import os
import sys

import stanza

input_file = sys.argv[1]
lang = sys.argv[2]
output = sys.argv[3]

with open(input_file, 'r') as infile:
    raw_text = infile.read().strip()

stanza.download(lang)
nlp = stanza.Pipeline(lang, use_gpu=True, processors='tokenize, lemma, pos, depparse',)

doc = nlp(raw_text)

with open(output, 'w') as outfile:
    for i, sentence in enumerate(doc.sentences):
        for token in sentence.tokens:
            for word in token.to_dict():
                line = f'{word["id"]}'
                for key in ['text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']:
                    try:
                        line = line + f'\t{word[key]}'
                    except KeyError:
                        line = line + f'\t_'
                line = line + "\t_\t_\n"
                outfile.write(line)
        outfile.write("\n")
        if (i > 0) and (i % 1000) == 0:
            print(f"Processed {i} sentences.")