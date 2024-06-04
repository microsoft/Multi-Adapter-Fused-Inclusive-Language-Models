#!/bin/bash
for bias in gender_adele gender_v2 profession race religion;
do
    for tokenizer in bert-base-multilingual-uncased xlm-roberta-base;
    do
        python tokenize_lm_corpus.py \
            --txt_file /home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/training_data/cda/$bias/raw.txt \
            --tokenizer_variant $tokenizer
    done
done