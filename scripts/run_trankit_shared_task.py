import logging
import os
import sys
import subprocess

from trankit import Pipeline

blind_data_dir = sys.argv[1]
save_dir = sys.argv[2] # 'saved_models'
output_dir = sys.argv[3]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

TBID_TO_LANG={
    "ar_padt": "ar",
    "bg_btb": "bg",
    "en_ewt": "en", #2
    "fi_tdt": "fi",
    "fr_sequoia": "fr",
    "it_isdt": "it",
    "lt_alksnis": "lt",
    "lv_lvtb": "lv",
    "cs_pdt": "cs", #3
    "et_ewt": "et", #2
    "pl_pdb": "pl",
    "nl_alpino": "nl",
    "ru_syntagrus": "ru",
    "sk_snk": "sk",
    "sv_talbanken": "sv",
    "ta_ttb": "ta",
    "uk_iu": "uk",
}

ISO_TO_TBID={
    "ar": "ar_padt",
    "bg": "bg_btb",
    "en": "en_ewt",
    "fi": "fi_tdt",
    "fr": "fr_sequoia",
    "it": "it_isdt",
    "lt": "lt_alksnis",
    "lv": "lv_lvtb",
    "cs": "cs_pdt",
    "et": "et_edt",
    "pl": "pl_pdb",
    "nl": "nl_alpino",
    "ru": "ru_syntagrus",
    "sk": "sk_snk",
    "sv": "sv_talbanken",
    "ta": "ta_ttb",
    "uk": "uk_iu",
}

ISO_TO_LONG={
    "ar":"arabic",
    "bg": "bulgarian",
    "cs": "czech",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "it": "italian",
    "lt": "lithuanian",
    "lv": "latvian",
    "nl": "dutch",
    "pl": "polish",
    "ru": "russian",
    "sk": "slovak",
    "sv": "slovenian",
    "uk": "ukrainian",
    "ta": "tamil",
}

DATA_DIR="../data/train-dev"
UD_TOOLS_DIR="../tools"

files = ["ar.txt", "ta.txt"]
#files  = ["ar.txt", "bg.txt", "cs.txt", "en.txt", "et.txt", "fi.txt", "fr.txt", "it.txt", "lv.txt", "lt.txt", "nl.txt", "pl.txt", "ru.txt", "sk.txt", "sv.txt", "uk.txt", "ta.txt"]

for file in files:
    iso = file.split(".")[0]
    tb_code = ISO_TO_TBID[iso]
    language = ISO_TO_LONG[iso]

    print(f"Running {iso}, {tb_code}, {language}")

    p = Pipeline(lang=language, gpu=True, cache_dir='./cache')
        
    with open(f'{blind_data_dir}/{iso}.txt', 'r') as infile:
        raw_text = infile.read().strip()
    infile.close()

    doc = p(raw_text)

    dummy_edep = "0:root"

    with open(f'{output_dir}/{iso}.conllu', 'w') as outfile:
        for i in range(len(doc['sentences'])):
            sentence_object = doc['sentences'][i]['tokens']
            for token_id in range(len(sentence_object)):
                token = sentence_object[token_id]

                skip_mwt = False
                # handle IDs first
                # The syntactic words are organized into a list stored in the field 'expanded'
                # https://trankit.readthedocs.io/en/latest/tokenize.html
                is_mwt = type(token["id"]) == tuple
                if is_mwt:
                    b_e = token["id"]
                    b = b_e[0]
                    e = b_e[1]

                    # if it's a 1 word MWT; skip the mwt and just write the expanded lines
                    if b == e:
                        # don't write 'line' at the end
                        skip_mwt = True

                    line = "-".join([str(b), str(e)])

                    # annotate the expanded syntactic tokens
                    expanded = token["expanded"]

                    tmp = []
                    for syntactic_token in expanded:
                        # clear the line for each syntactic token
                        expanded_line = ""
                        expanded_line = str(syntactic_token["id"])
                        for key in ['text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']:
                            try:
                                feature = f"\t{syntactic_token[key]}"
                                expanded_line = expanded_line + str(feature)
                            except KeyError:
                                # not all langs have 'xpos'
                                expanded_line = expanded_line + f"\t_" # empty line
                        expanded_line = expanded_line + f"\t{dummy_edep}\t_\n"
                        
                        ## return these expanded lines
                        tmp.append(expanded_line)
                   
                    expanded_lines = ""
                    for s in tmp:
                        expanded_lines = expanded_lines + s

                else:
                    line = str(token["id"])

                for key in ['text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel']: # missing deps, misc
                    try:
                        feature = f"\t{token[key]}"
                        line = line + str(feature)
                    except KeyError:
                        # not all langs have 'xpos'
                        line = line + f"\t_" # empty line

                # empty values for deps and misc.
                if not is_mwt:
                    line = line + f"\t{dummy_edep}\t_\n"
                else:
                    # write mwt and syntactic tokens
                    if not skip_mwt:
                        line = line + "\t_\t_\n"
                        line = line + expanded_lines
                    # write just the syntactic tokens to avoid (b=e) spans.
                    else:
                        line = expanded_lines
                    
                outfile.write(line)

            outfile.write("\n")
            if (i > 0) and (i % 1000) == 0:
                print(f"Processed {i} sentences.")
