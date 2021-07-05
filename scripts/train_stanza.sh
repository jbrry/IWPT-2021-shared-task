#!/bin/bash

test -z $1 && echo "Missing list of TBIDs (space or colon-separated), e.g. 'fr_sequoia'"
test -z $1 && exit 1
TBIDS=$(echo $1 | tr ':' ' ')

DATA_DIR="data/ud-treebanks-v2.7"

if [ ! -s "$DATA_DIR" ]; then
    echo "missing data directory!"
    echo "cd to stanza directory and run the script in scripts/download_ud_data.sh"
fi

source scripts/config.sh

# TBIDS="ar_padt nl_lassysmall fr_sequoia ru_syntagrus bg_btb en_ewt it_isdt sk_snk cs_cac en_gum lv_lvtb sv_talbanken
# cs_fictree et_edt lt_alksnis ta_ttb cs_pdt et_ewt  pl_lfg uk_iu nl_alpino fi_tdt pl_pdb" 

for TBID in $TBIDS; do
    echo == $TBID ==
    for processor in tokenizer mwt pos lemma depparse; do
        echo "Running $processor"

        for filepath in ${DATA_DIR}/*/${TBID}-ud-train.conllu; do
            dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
            TREEBANK=`basename $dir`       # e.g. UD_Afrikaans-AfriBooms
            
            python stanza/utils/datasets/prepare_${processor}_treebank.py ${TREEBANK}
            sleep 1
            python stanza/utils/training/run_${processor}.py ${TREEBANK}

        done
    done
done
