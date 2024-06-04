#!/bin/bash

for bias in gender_v2 gender_adele profession race religion;
do    
    python pretrain_dba.py \
        --data-dir "/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/training_data/cda/$bias/tokenized/" \
        --restore-from "/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts/debiasing/$bias" \
        --out-dir "/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts/debiasing/$bias" \
        --model-base-model bert-base-multilingual-uncased \
        --model-tokenizer bert-base-multilingual-uncased \
        --model-adapter-name "debiasing_$bias" \
        --do-eval
done