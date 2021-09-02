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

#weights=(/home/ubuntu/finBERT/weights/resume_decay0.001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1400.bin /home/ubuntu/finBERT/weights/resume_decay0.0001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1400.bin /home/ubuntu/finBERT/weights/resume_decay0.001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1200.bin /home/ubuntu/finBERT/weights/resume_decay0.0001_lr5e-6_ss1024_bs16_results_finbert/pytorch_model_1200.bin)
#names=(resume_decay0.001_lr5e-6_1400 resume_decay0.0001_lr5e-6_1400 resume_decay0.001_lr5e-6_1200 resume_decay0.0001_lr5e-6_1200)

weights=(/home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt1.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt2.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt6.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd0.5_ckpt1.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd0.5_ckpt2.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd0.5_ckpt10.bin)
names=(decay0.5_lr1e-4_ckpt1 decay0.5_lr1e-4_ckpt2 decay0.5_lr1e-4_ckpt6 decay0.5_lr1e-5_ckpt1 decay0.5_lr1e-5_ckpt2 decay0.5_lr1e-5_ckpt10)
seeds=(42 125380 160800 22758 176060 193228)

# bsssconfigs=(x y z)
# declare -A batch_sizes=(
#     [x]=4
# )
#batch_sizes=(4 )
max_lengths=(60 1024)

#lr=5e-5
#wd=0.001

bs=4
#max_length=1024

lrwdconfigs=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
declare -A lrs=(
    [1]=1e-4
    [2]=1e-4
    [3]=1e-4
    [4]=1e-4
    [5]=1e-5
    [6]=1e-5
    [7]=1e-5
    [8]=1e-5
    [9]=5e-5
    [10]=5e-5
    [11]=5e-5
    [12]=5e-5
    [13]=1e-6
    [14]=1e-6
    [15]=1e-6
    [16]=1e-6
)
declare -A wds=(
    [1]=0.01
    [2]=0.1
    [3]=0.5
    [4]=1.0
    [5]=0.01
    [6]=0.1
    [7]=0.5
    [8]=1.0
    [9]=0.01
    [10]=0.1
    [11]=0.5
    [12]=1.0
    [13]=0.01
    [14]=0.1
    [15]=0.5
    [16]=1.0
)
for i in ${!weights[@]}; do
    for seed in ${seeds[@]}; do
        for max_length in ${max_lengths[@]}; do
            for lrwdconfig in ${lrwdconfigs[@]}; do
                lr=${lrs[$lrwdconfig]}
                wd=${wds[$lrwdconfig]}
                start=`date +%s`
                weight=${weights[i]}
                name=${names[i]}
                echo "name: ${name}, weight: ${weight}"
                out_folder=/home/ubuntu/finBERT/gpt_downstream/tadp_eval_gridsearch/${name}
                [ -d ${out_folder} ] || mkdir -p ${out_folder}
                log_file=/home/ubuntu/finBERT/gpt_downstream/tadp_eval_gridsearch/${name}/full_name_${name}_seed_${seed}_bs_${bs}_ss_${max_length}.txt
                # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch
                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch example.py --name ${name} --weight ${weight} --lr ${lr} --wd ${wd} --seed ${seed} --bs ${bs} --max_length ${max_length} 2>&1 | tee ${log_file}
                end=`date +%s`
                runtime=$((end-start))
                echo "time taken: ${runtime}"
            done
        done
    done
done