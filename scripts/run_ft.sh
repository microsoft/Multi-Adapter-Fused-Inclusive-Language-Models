#!/bin/bash

for task in stsb mnli;
do
    for bias in religion race profession gender;
    do
        python src/finetune_on_downstream_task.py \
            --model_name bert-base-multilingual-cased \
            --tokenizer_name bert-base-multilingual-cased \
            --save_to /home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts/finetuning/${task}/ta_without_${bias}_dba/ \
            --task_name $task \
            --add_task_adapter
        
        python src/finetune_on_downstream_task.py \
            --model_name bert-base-multilingual-cased \
            --tokenizer_name bert-base-multilingual-cased \
            --dba_dir /home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts/debiasing/$bias \
            --save_to /home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts/finetuning/${task}/ta_with_${bias}_dba/ \
            --task_name $task \
            --add_debiasing_adapter \
            --add_task_adapter
    done
done