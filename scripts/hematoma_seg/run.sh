#!/bin/bash

#cd ../..

export HF_ENDPOINT=https://hf-mirror.com

# custom config
DATA="/dingshaodong/projects/hematoma_meta/data/preprocessed/resampled"
TRAINER=HematomaSeg
DATASET=hematoma
SEED=$1

CFG=dyunet_ep50
# SHOTS=16

# I need the shell script to take a list of source domains.
# Define the list of strings
SOURCE_DOMAINS=("high_quality")
# Join the elements of the list with "_" using parameter expansion
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS[*]}"
SOURCE_DOMAINS_STR="${SOURCE_DOMAINS_STR// /_}"  # Replace spaces with underscores

DIR=output/${DATASET}/${TRAINER}/${CFG}/${SOURCE_DOMAINS_STR}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/uda/datasets/${DATASET}.yaml \
    --config-file configs/uda/trainers/${TRAINER}/${CFG}.yaml \
    --source-domains ${SOURCE_DOMAINS[*]} \
    --output-dir ${DIR} \
fi