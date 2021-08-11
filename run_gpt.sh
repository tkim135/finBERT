#!/bin/bash

# for folder in ${myArray[@]}; do
#     echo "${folder}"
#     run_folder=$(echo "${folder}/" | cut -d'/' -f8)
#     echo "${run_folder}"
#     log_file="downstream_eval_gridsearch_results/pile/downstreamgrid${run_folder}.txt"
#     echo "${log_file}"
#     CUDA_VISIBLE_DEVICES=1 python3 -u run_seeds.py --model_name_or_path ${folder} 2>&1 --start_seed 0 --end_seed 4 --type pile | tee ${log_file}
#     rm -rf /scratch/varunt/finbert_clpath/pile/
# done

learning_rates=(5e-5 5e-4 5e-6 1e-7 5e-7)
weight_decays=(0.001 0.01 0.0001 0.005 0.0005)
seeds=(42 125380 160800 22758 176060 193228)
for lr in ${learning_rates[@]}; do
    for wd in ${weight_decays[@]}; do
        for seed in ${seeds[@]}; do
            start=`date +%s`
            log_file=/home/ubuntu/finBERT/gpt_downstream/config_gridsearch/full_log_${lr}_${wd}_${seed}.txt
            #CUDA_VISIBLE_DEVICES=0 python3 -u run_seeds.py --model_name_or_path "gpt2" 2>&1 --start_seed 0 --end_seed 2 --type public | tee ${log_file}
            CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch example.py --lr ${lr} --wd ${wd} --seed ${seed} 2>&1 | tee ${log_file}
            end=`date +%s`
            runtime=$((end-start))
            echo "time taken: ${runtime}"
        done
    done
done