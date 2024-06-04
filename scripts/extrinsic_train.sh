#!/bin/bash 

# add all the commands you want to run here

# task name (stsb / mnli)
task=$1

for type in gender race religion; do 
    python finetune_on_downstream_task.py \
        --save_to ../checkpoints_v2/${type}/${task}_ta \
        --task_name ${task} \
        --add_task_adapter 

    python finetune_on_downstream_task.py \
        --save_to ../checkpoints_v2/${type}/${task}_with_dba_ta \
        --dba_dir ../checkpoints_v2/${type}/debiasing_adapter_checkpoint \
        --add_debiasing_adapter \
        --add_task_adapter \
        --task_name ${task}
done

python finetune_on_downstream_task.py \
    --dba_dir ../checkpoints_v2/gender/debiasing_adapter_checkpoint:../checkpoints_v2/race/debiasing_adapter_checkpoint:../checkpoints_v2/religion/debiasing_adapter_checkpoint:../checkpoints_v2/profession/debiasing_adapter_checkpoint \
    --fusion_dir ../checkpoints_v2/fusion/debiasing_adapter_checkpoint \
    --save_to ../checkpoints_v2/fusion/${task}_with_fusion_ta \
    --add_debiasing_adapter \
    --add_task_adapter \
    --task_name ${task}

