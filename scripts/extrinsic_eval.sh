#!/bin/bash 

task=$1

for type in gender race religion; do 

    python compute_sim_scores.py \
        --infile ../checkpoints_v2/${type}/${task}_ta/similarity_scores.csv \
        --outfile ../checkpoints_v2/${type}/${task}_ta/results.json \
        --task_name ${task}

    python compute_sim_scores.py \
        --infile ../checkpoints_v2/${type}/${task}_with_dba_ta/similarity_scores.csv \
        --outfile ../checkpoints_v2/${type}/${task}_with_dba_ta/results.json \
        --task_name ${task}
done