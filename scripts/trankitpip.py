from trankit import Pipeline
import sys

input_file = sys.argv[1]
lang = sys.argv[2]
output = sys.argv[3]

sents = []
with open(input_file) as f:
	for line in f:
		sents.append(line.split())
    
print(sents[0:10])

p = Pipeline(lang)
tagged_doc = p.posdep(sents)


dummy_edep = "0:root"

with open(output,'w') as outfile:
    for sent in tagged_doc['sentences']:
        for token in sent['tokens']:
            line = str(token["id"])

            for key in ['text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']: # missing deps, misc
                try:
                    feature = f"\t{token[key]}"
                    line = line + str(feature)
                except KeyError:
                    # not all langs have 'xpos'
                    line = line + f"\t_" # empty line

            # deps and misc
            line = line + f"\t{dummy_edep}\t_\n"
            outfile.write(line)
        outfile.write("\n")

