#!/bin/bash

export bias="religion"

python pretrain_dba.py \
    --data-dir "/home/t-assathe/BlobStorage/adapter-debiasing/training_data/cda/$bias/tokenized/" \
    --model-base-model bert-base-multilingual-uncased \
    --model-tokenizer bert-base-multilingual-uncased \
    --do-eval