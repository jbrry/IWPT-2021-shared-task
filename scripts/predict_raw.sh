#!/bin/bash

# ./scripts/predict_raw.sh logs/iwpt2021-submission-no_delex/en_ewt-transformer-enhanced_kg_parser-transformer-no_delex-756432-20230719-055151 input.txt multitask_parser

test -z $1 && echo "Missing model"
test -z $1 && exit 1
MODEL=$1

test -z $2 && echo "Missing input file"
test -z $2 && exit 1
INPUT_FILE=$2

test -z $3 && echo "Missing package version <multitask_parser>"
test -z $3 && exit 1
PACKAGE=$3

echo "using $PACKAGE"

LANGUAGE="en"

FILENAME=${INPUT_FILE%.*}  # Extracting the filename without extension
TMP_OUTPUT=${FILENAME}.conllu
FINAL_OUTPUT=${FILENAME}-eud.conllu

python scripts/run_stanza.py ${INPUT_FILE} ${LANGUAGE} ${TMP_OUTPUT}

#=== Predict ===
allennlp predict ${MODEL}/model.tar.gz ${TMP_OUTPUT} \
    --output-file ${FINAL_OUTPUT} \
    --predictor enhanced-predictor \
    --include-package "$PACKAGE" \
    --use-dataset-reader \
    --silent
