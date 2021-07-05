#!/bin/bash

# https://trankit.readthedocs.io/en/latest/training.html

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
PROJECT_DIRECTORY="${DIR%/*}"

# Download UD tools directory (to check well-formedness of outputs).
if [ ! -s "$PROJECT_DIRECTORY/tools" ]; then
    echo "missing UD tools, downloading to $PROJECT_DIRECTORY ..."
    git clone https://github.com/UniversalDependencies/tools.git
fi

DATA_DIR="data/train-dev"
TRANKIT_RUN="/home/jbarry/spinning-storage/jbarry/trankit/prepare_test_IWPT.py"

UD_TOOLS_DIR="tools"

if [ ! -s "$DATA_DIR" ]; then
    echo "missing data directory!"
    echo "run: ./scripts/download_iwpt_data.sh to download shared task data."
fi

TBIDS="ar_padt nl_lassysmall fr_sequoia ru_syntagrus bg_btb en_ewt it_isdt sk_snk cs_cac en_gum lv_lvtb sv_talbanken
cs_fictree et_edt lt_alksnis ta_ttb cs_pdt et_ewt  pl_lfg uk_iu nl_alpino fi_tdt pl_pdb" 


for TBID in $TBIDS; do
    echo == $TBID ==
    echo "Running Trankit..."

    for TASK in tokenize posdep lemmatize; do

        for filepath in ${DATA_DIR}/*/${TBID}-ud-train.conllu; do
            dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
            TREEBANK=`basename $dir`       # e.g. UD_Afrikaans-AfriBooms

            TRAIN_TXT="${DATA_DIR}/${TREEBANK}/${TBID}-ud-train.txt"
            TRAIN_CONLLU="${DATA_DIR}/${TREEBANK}/${TBID}-ud-train.conllu"
            DEV_TXT="${DATA_DIR}/${TREEBANK}/${TBID}-ud-dev.txt"
            DEV_CONLLU="${DATA_DIR}/${TREEBANK}/${TBID}-ud-dev.conllu"
            
            python $TRANKIT_RUN --train-txt $TRAIN_TXT \
		    --train-conllu $TRAIN_CONLLU \
		    --dev-txt $DEV_TXT \
		    --dev-conllu $DEV_CONLLU \
		    --task $TASK \
		    --save-dir logs/trankit/$TBID \
		    --train \
		    --predict

        done
    done
done
