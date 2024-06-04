#!/bin/bash

for bias in gender_v2 profession race religion;
do
    python src/pretrain_dba.py \
        --data-dir "/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/training_data/cda/$bias/tokenized/" \
        --out-dir "/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts/debiasing/$bias" \
        --model-base-model bert-base-multilingual-uncased \
        --model-tokenizer bert-base-multilingual-uncased \
        --model-adapter-name "debiasing_$bias" \
        --model-non-linearity "silu" \
        --training-do-train True \
        --training-per-device-train-batch-size 352 \
        --training-gradient-accumulation-steps 4 \
        --training-num-train-epochs 2 \
        --training-learning-rate "3e-5" 
done