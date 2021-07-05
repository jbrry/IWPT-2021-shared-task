#!/usr/bin/env bash

# adapted from: https://github.com/Hyperparticle/udify/blob/master/scripts/concat_ud_data.sh

# czech
# ./scripts/concat_ud_data.sh UD_Czech-PDT:UD_Czech-CAC:UD_Czech-FicTree

# english
# ./scripts/concat_ud_data.sh UD_English-EWT:UD_English-GUM

# dutch
# ./scripts/concat_ud_data.sh UD_Dutch-Alpino:UD_Dutch-LassySmall

# estonian
# ./scripts/concat_ud_data.sh UD_Estonian-EDT:UD_Estonian-EWT

# polish
# ./scripts/concat_ud_data.sh UD_Polish-PDB:UD_Polish-LFG

test -z $1 && echo "Missing list of Treebanks (space or colon-separated), e.g. 'UD_English-EWT:UD_English-GUM"
test -z $1 && exit 1
TREEBANKS=$(echo $1 | tr ':' ' ')

# IWPT data dir
DATASET_DIR="data/train-dev"

echo "Generating concatenated dataset..."

if [[ ! -d "data/train-dev-concatenated" ]]; then
    # start with default files
    echo "Copying regular files..."
    cp -r "data/train-dev" "data/train-dev-concatenated"
fi

python scripts/concat_treebanks.py "data/train-dev-concatenated" --dataset_dir ${DATASET_DIR} --treebanks ${TREEBANKS}