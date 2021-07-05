#!/bin/bash

# sorted metadata: [
#     ('cs_pdt-ud-train.conllu', [68495, 2140]), ('ru_syntagrus-ud-train.conllu', [48814, 1525]), ('et_edt-ud-train.conllu', [24633, 770]), ('cs_cac-ud-train.conllu', [23478, 734]),
#     ('pl_pdb-ud-train.conllu', [17722, 554]), ('pl_lfg-ud-train.conllu', [13774, 430]), ('it_isdt-ud-train.conllu', [13121, 410]), ('en_ewt-ud-train.conllu', [12543, 392]),
#     ('nl_alpino-ud-train.conllu', [12264, 383]), ('fi_tdt-ud-train.conllu', [12217, 382]), ('cs_fictree-ud-train.conllu', [10160, 318]), ('lv_lvtb-ud-train.conllu', [10156, 317]),
#     ('bg_btb-ud-train.conllu', [8907, 278]), ('sk_snk-ud-train.conllu', [8483, 265]), ('ar_padt-ud-train.conllu', [6075, 190]), ('nl_lassysmall-ud-train.conllu', [5787, 181]),
#     ('en_gum-ud-train.conllu', [5670, 177]), ('uk_iu-ud-train.conllu', [5496, 172]), ('sv_talbanken-ud-train.conllu', [4303, 134]), ('et_ewt-ud-train.conllu', [2837, 89]),
#     ('lt_alksnis-ud-train.conllu', [2341, 73]), ('fr_sequoia-ud-train.conllu', [2231, 70]), ('ta_ttb-ud-train.conllu', [400, 12])]

# Delexicalised data
test -z $1 && echo "Missing data type: 'no_delex', 'delex'"
test -z $1 && exit 1
DELEX_TYPE=$1


# some high memory tbids
HIGH_MEMORY_TBIDS=(
    "ar_padt"
    "cs_cac"
    "ru_syntagrus"
    "cs_pdt"
    "et_edt"
    "pl_pdb"
)

MEDIUM_TBIDS=(
    "lv_lvtb"
    "cs_fictree"
    "fi_tdt"
    "nl_alpino"
    "en_ewt"
    "it_isdt"
    "pl_lfg"
)

# if any of these have OOM issue, move to HIGH_MEMORY_TBIDS
# < 10k train sentences
SHORT_TBIDS=(
    "ta_ttb"
    "fr_sequoia"
    "lt_alksnis"
    "et_ewt"
    "sv_talbanken"
    "uk_iu"
    "en_gum"
    "nl_lassysmall"
    "sk_snk"
    "bg_btb"
)

# just use XLM-R for the moment.
MODEL="xlm-roberta-base"

# run high memory jobs first
echo "running high memory tbids"
for ((i=0;i<${#HIGH_MEMORY_TBIDS[@]};++i)); do
    TBID=${HIGH_MEMORY_TBIDS[i]}
    echo
    echo "== running ${TBID} with ${MODEL} =="

    # NOTE: Make sure to add your cluster options here, e.g.
    # GPUS: --gres=gpu:rtx6000:1, --gres=gpu:rtx2080ti:1 or --gres=gpu:v100:1 (if you have one)
    # cluster partition: -p long (long job), -p compute (<2 days) - this will be different depending on your cluster
    
    # an example one would look like:  
    # sbatch -p long --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh en_ewt enhanced transformer xlm-roberta-base enhanced_kg_parser transformer
    sbatch -J eud_high_mem --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh ${TBID} enhanced transformer ${MODEL} enhanced_kg_parser transformer ${DELEX_TYPE}
done

echo "running medium tbids"
for ((i=0;i<${#MEDIUM_TBIDS[@]};++i)); do
    TBID=${MEDIUM_TBIDS[i]}
    echo
    echo "== running ${TBID} with ${MODEL} =="

    # sbatch -p long --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh en_ewt enhanced transformer xlm-roberta-base enhanced_kg_parser transformer
    sbatch -J eud_medium --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh ${TBID} enhanced transformer ${MODEL} enhanced_kg_parser transformer ${DELEX_TYPE}
done

echo "running short tbids"
for ((i=0;i<${#SHORT_TBIDS[@]};++i)); do
    TBID=${SHORT_TBIDS[i]}
    echo
    echo "== running ${TBID} with ${MODEL} =="

    # sbatch -p long --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh en_ewt enhanced transformer xlm-roberta-base enhanced_kg_parser transformer # 
    sbatch -J eud_small --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh ${TBID} enhanced transformer ${MODEL} enhanced_kg_parser transformer ${DELEX_TYPE}
done