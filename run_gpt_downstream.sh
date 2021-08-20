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

#weights=(/home/ubuntu/finBERT/weights/decay0.0001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model.bin /home/ubuntu/finBERT/weights/decay0.001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model.bin)
#/home/ubuntu/finBERT/weights/resume_decay0.001_lr5e-6_ss1024_bs16_results_finbert
#/home/ubuntu/finBERT/weights/resume_decay0.0001_lr5e-6_ss1024_bs16_results_finbert
#pytorch_model_1200.bin
#pytorch_model_1400.bin
weights=(/home/ubuntu/finBERT/weights/resume_decay0.001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1400.bin /home/ubuntu/finBERT/weights/resume_decay0.0001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1400.bin /home/ubuntu/finBERT/weights/resume_decay0.001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1200.bin /home/ubuntu/finBERT/weights/resume_decay0.0001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1200.bin)
names=(resume_decay0.001_lr5e-6_1400 resume_decay0.0001_lr5e-6_1400 resume_decay0.001_lr5e-6_1200 resume_decay0.0001_lr5e-6_1200)
seeds=(42 125380 160800 22758 176060 193228)
lr=5e-5
wd=0.001
for i in ${!weights[@]}; do
    for seed in ${seeds[@]}; do
        start=`date +%s`
        weight=${weights[i]}
        name=${names[i]}
        echo "name: ${name}, weight: ${weight}"
        out_folder=/home/ubuntu/finBERT/gpt_downstream/eval_gridsearch/${name}
        [ -d ${out_folder} ] || mkdir -p ${out_folder}
        log_file=/home/ubuntu/finBERT/gpt_downstream/eval_gridsearch/${name}/full_name_${name}_${seed}.txt
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch example.py --name ${name} --weight ${weight} --lr ${lr} --wd ${wd} --seed ${seed} 2>&1 | tee ${log_file}
        end=`date +%s`
        runtime=$((end-start))
        echo "time taken: ${runtime}"
    done
done