#!/bin/bash

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: James Barry

# sample usage: ./scripts/predict.sh fr_sequoia logs/iwpt2021-submission-delex/fr_sequoia-transformer-enhanced_kg_parser-transformer-delex-756432-20210520-200145 multitask_parser

test -z $1 && echo "Missing TBID"
test -z $1 && exit 1
TBID=$1

test -z $2 && echo "Missing model"
test -z $2 && exit 1
MODEL=$2

test -z $3 && echo "Missing package version <multitask_parser>"
test -z $3 && exit 1
PACKAGE=$3

echo "using $PACKAGE"

mkdir -p output

# the following requires https://github.com/UniversalDependencies/tools to be downloaded on your system 
# and some possible perl requirements, see: https://github.com/Jbar-ry/Enhanced-UD-Parsing/issues/1 

# official shared-task data
TB_DIR=data/train-dev

UD_TOOLS_DIR=tools

echo "== $TBID =="

for filepath in ${TB_DIR}/*/${TBID}-ud-train.conllu; do
  dir=`dirname $filepath`        # e.g. /home/user/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms
  tb_name=`basename $dir`        # e.g. UD_Afrikaans-AfriBooms

  GOLD=${TB_DIR}/${tb_name}/${TBID}-ud-dev.conllu
  PRED=output/${TBID}_pred.conllu

  #=== Predict ===
  allennlp predict ${MODEL}/model.tar.gz ${GOLD} \
    --output-file ${PRED} \
    --predictor enhanced-predictor \
    --include-package "$PACKAGE" \
    --use-dataset-reader \
    --silent

  # collapse empty nodes in gold file
  perl ${UD_TOOLS_DIR}/enhanced_collapse_empty_nodes.pl ${GOLD} > output/${tbid}_gold_collapsed.conllu

  # collapse empty nodes in pred file
  perl ${UD_TOOLS_DIR}/enhanced_collapse_empty_nodes.pl ${PRED} > output/${tbid}_pred_collapsed.conllu

  echo "Running UD Shared Task evaluation script"
  python scripts/iwpt21_xud_eval.py --verbose output/${tbid}_gold_collapsed.conllu output/${tbid}_pred_collapsed.conllu > output/${tbid}_pred.result

done
