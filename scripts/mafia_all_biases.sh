#!/bin/bash

export DEBIASING_ROOT="/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/ckpts_Bert/"
export FUSION_CKPT_ROOT="/home/t-assathe/local_ckpts_stacked_bert/" 
# It's better to store the ckpts locally because the JSON for the best model need not be necessarily refreshed on the Azure container

task_name="jigsaw"

export TOKENIZERS_PARALLELISM=false
# https://stackoverflow.com/a/32641190
items=(gender race religion profession)
n=${#items[@]}
powersize=$((1 << $n))

i=0
while [ $i -lt $powersize ]
do
    subset=()
    j=0
    while [ $j -lt $n ]
    do
        if [ $(((1 << $j) & $i)) -gt 0 ]
        then
            subset+=("${items[$j]}")
        fi
        j=$(($j + 1))
    done
    echo "'${subset[@]}'"
    folder_name="${subset[0]}"
    fusion_paths="${DEBIASING_ROOT}/${subset[0]}"
    m=${#subset[@]}
    for (( k=1 ; k<$m ; k++));
    do
        folder_name="${folder_name}+${subset[k]}"
        fusion_paths="${fusion_paths} ${DEBIASING_ROOT}/${subset[k]}"
    done
    WANDB_PROJECT="ModularDebiasingMSR" WANDB_NAME="fusion:$folder_name:$task_name" python src/minimal_task_ft.py \
        --save_dir_fusion "${FUSION_CKPT_ROOT}/fusion_${task_name}/${folder_name}" \
        --debiasing_adapter_paths $fusion_paths \
        --expt_name "fusion:$folder_name" \
        --model_name bert-base-uncased \
        --tokenizer_name bert-base-uncased \
        --task_name $task_name \
        --add_task_adapter True \
        --task_adapter_type pfeiffer \
        --task_adapter_activation silu \
        --block_size 128 \
        --metric_for_best_model eval_accuracy \
        --greater_is_better \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --remove_unused_columns False \
        --label_names labels \
        --fp16 \
        --load_best_model_at_end True \
        --overwrite_output_dir True \
        --save_total_limit 1 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 512 \
        --per_device_eval_batch_size 4096 \
        --logging_steps 500 \
        --weight_decay 0.01 \
        --learning_rate "1e-5" \
        --optim "adamw_torch" \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.1 \
        --report_to tensorboard
    i=$(($i + 1))
done