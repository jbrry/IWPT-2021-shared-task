#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
PROJECT_DIRECTORY="${DIR%/*}"

# Download delexicalisation tool.
# Note: This is still a work in progress.
if [ ! -s "$PROJECT_DIRECTORY/conllugraph" ]; then
    echo "missing conllugraph, downloading to $PROJECT_DIRECTORY ..."
    git clone https://github.com/jbrry/conllugraph.git
fi

# Download UD tools directory.
if [ ! -s "$PROJECT_DIRECTORY/tools" ]; then
    echo "missing UD tools, downloading to $PROJECT_DIRECTORY ..."
    git clone https://github.com/UniversalDependencies/tools.git
fi

DATA_DIR="data/train-dev"
UD_TOOLS_DIR="tools"

if [ ! -s "$DATA_DIR" ]; then
    echo "missing data directory!"
    echo "run: ./scripts/download_iwpt_data.sh to download shared task data."
fi

TBIDS="ar_padt nl_lassysmall fr_sequoia ru_syntagrus bg_btb en_ewt it_isdt sk_snk cs_cac en_gum lv_lvtb sv_talbanken
cs_fictree et_edt lt_alksnis ta_ttb cs_pdt et_ewt pl_lfg uk_iu nl_alpino fi_tdt pl_pdb" 

echo "=== delexicalising =="

for TBID in $TBIDS; do
    echo == $TBID ==
    for filetype in train dev; do
        echo "Delexicalising $filetype files..."

        for filepath in ${DATA_DIR}/*/${TBID}-ud-train.conllu; do
            dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
            TREEBANK=`basename $dir`       # e.g. UD_Afrikaans-AfriBooms

            CONLLU_FILE="${DATA_DIR}/${TREEBANK}/${TBID}-ud-${filetype}.conllu"

            if [ -f ${CONLLU_FILE} ]; then
                python conllugraph/delexicalise_enhanced_dependencies.py -i "${CONLLU_FILE}"
            else
                echo "${CONLLU_FILE} not found, skipping..."
            fi

        done
    done
done

DELEX_DIR="${DATA_DIR}-delexicalised"
RELEX_DIR="${DATA_DIR}-delexicalised-relexicalised"

echo === relexicalising ===
for TBID in $TBIDS; do
    for filetype in train dev; do
        echo "Relexicalising $filetype files..."

        for filepath in ${DELEX_DIR}/*/${TBID}-ud-train.conllu; do
            dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
            TREEBANK=`basename $dir`       # e.g. UD_Afrikaans-AfriBooms

            CONLLU_FILE="${DELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}.conllu"
            if [ -f ${CONLLU_FILE} ]; then
                python conllugraph/relexicalise_enhanced_dependencies.py -i "${CONLLU_FILE}"
            else
                echo "${CONLLU_FILE} not found, skipping..."
            fi
        
            # compare results to gold
            GOLD="${DATA_DIR}/${TREEBANK}/${TBID}-ud-${filetype}.conllu"
            SYSTEM="${RELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}.conllu"

            # VALIDATE LVL 2
            LCODE=$(echo ${TBID} | awk -F "_" '{print $1}')
            cat ${SYSTEM} | python tools/validate.py --level 2 --lang ${LCODE}

            # collapse empty nodes in gold file
            # some extra perl requirements are required to work with the UD_TOOLS: see https://github.com/jbrry/IWPT-2021-shared-task/wiki/Installing-Perl-Dependencies
            perl ${UD_TOOLS_DIR}/enhanced_collapse_empty_nodes.pl ${GOLD} > "${RELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}_gold_collapsed.conllu"

            # collapse empty nodes in pred file
            perl ${UD_TOOLS_DIR}/enhanced_collapse_empty_nodes.pl ${SYSTEM} > "${RELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}_collapsed.conllu"

            echo "Running UD Shared Task evaluation script"
            python scripts/iwpt21_xud_eval.py --verbose "${RELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}_gold_collapsed.conllu" "${RELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}_collapsed.conllu" \
                > "${RELEX_DIR}/${TREEBANK}/${TBID}-ud-${filetype}_relexicalised.result"

        done
    done
done


# visualise output
cd ${RELEX_DIR}
rm delexicalised_scores.txt
grep "ELAS" */*.result | awk '{print $1 $3}' | sed 's/:/\t/' >> delexicalised_scores.txt

echo "Done"
