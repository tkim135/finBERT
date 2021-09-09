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

# downstream
#weights=(/home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-6_ss1024_bs16_results_finbert/pytorch_model_lr1.e-6_wd1.0_ckpt7.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-6_ss1024_bs16_results_finbert/pytorch_model_lr1.e-6_wd1.0_ckpt9.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt1.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt2.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt6.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd0.5_ckpt1.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd0.5_ckpt2.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd0.5_ckpt10.bin)
#names=(decay1.0_lr1e-6_ckpt7 decay1.0_lr1e-6_ckpt9 decay0.5_lr1e-4_ckpt1 decay0.5_lr1e-4_ckpt2 decay0.5_lr1e-4_ckpt6 decay0.5_lr1e-5_ckpt1 decay0.5_lr1e-5_ckpt2 decay0.5_lr1e-5_ckpt10)
#weights=(/home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-6_ss1024_bs16_results_finbert/pytorch_model_lr1.e-6_wd0.5_ckpt10.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd1.0_ckpt1.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd1.0_ckpt2.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd1.0_ckpt10.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd1.0_ckpt2.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay1.0_lr1.e-5_ss1024_bs16_results_finbert/pytorch_model_lr1.e-5_wd1.0_ckpt3.bin public_ckpt)
#names=(decay0.5_lr1e-6_ckpt10 decay1.0_lr1e-4_ckpt1 decay1.0_lr1e-4_ckpt2 decay1.0_lr1e-5_ckpt10 decay1.0_lr1e-5_ckpt2 decay1.0_lr1e-5_ckpt3 hf_public_ckpt)

# public
# weights=(public_ckpt /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt6.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt6.bin)
# names=(hf_public_ckpt decay0.5_lr1e-4_ckpt6 decay0.5_lr1e-4_ckpt6)
# small_vocab=("True" "False" "True")
weights=(/home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt6.bin /home/ubuntu/finBERT/weights/hf_ckpt_decay0.5_lr1.e-4_ss1024_bs16_results_finbert/pytorch_model_lr1.e-4_wd0.5_ckpt6.bin)
names=(decay0.5_lr1e-4_ckpt6 decay0.5_lr1e-4_ckpt6)
small_vocab=("False" "True")

seeds=(42 125380 160800 22758 176060 193228)

# bsssconfigs=(x y z)
# declare -A batch_sizes=(
#     [x]=4
# )
#batch_sizes=(4 )
max_lengths=(60)

#lr=5e-5
#wd=0.001

bs=4
#max_length=1024

# lrwdconfigs=(a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 c1 c2 c3 c4 c5)
# declare -A lrs=(
#     [a1]=5e-5
#     [a2]=5e-5
#     [a3]=5e-5
#     [a4]=5e-5
#     [a5]=5e-5
#     [b1]=1e-5
#     [b2]=1e-5
#     [b3]=1e-5
#     [b4]=1e-5
#     [b5]=1e-5
#     [c1]=1e-6
#     [c2]=1e-6
#     [c3]=1e-6
#     [c4]=1e-6
#     [c5]=1e-6
# )
# declare -A wds=(
#     [a1]=0.001
#     [a2]=0.01
#     [a3]=0.1
#     [a4]=0.5
#     [a5]=1.0
#     [b1]=0.001
#     [b2]=0.01
#     [b3]=0.1
#     [b4]=0.5
#     [b5]=1.0
#     [c1]=0.001
#     [c2]=0.01
#     [c3]=0.1
#     [c4]=0.5
#     [c5]=1.0
# )

lrwdconfigs=(a1)
declare -A lrs=(
    [a1]=5e-5
)
declare -A wds=(
    [a1]=0.001
)

for max_length in ${max_lengths[@]}; do
    for lrwdconfig in ${lrwdconfigs[@]}; do
        for i in ${!weights[@]}; do
            for seed in ${seeds[@]}; do
                lr=${lrs[$lrwdconfig]}
                wd=${wds[$lrwdconfig]}
                start=`date +%s`
                weight=${weights[i]}
                name=${names[i]}
                use_smaller_vocab=${small_vocab[i]}
                gradual_unfreeze="False"
                discriminate="False"
                echo "name: ${name}, weight: ${weight}"
                out_folder=/home/ubuntu/finBERT/gpt_downstream/tadp_eval_gridsearch/${name}
                #out_folder=/home/ubuntu/finBERT/public_gridsearch/tadp_eval_gridsearch/${name}
                [ -d ${out_folder} ] || mkdir -p ${out_folder}
                log_file=/home/ubuntu/finBERT/gpt_downstream/tadp_eval_gridsearch/${name}/full_name_${name}_seed_${seed}_bs_${bs}_ss_${max_length}_ftlr_${lr}_ftwd_${wd}.txt
                #log_file=/home/ubuntu/finBERT/gpt_downstream/public_gridsearch/${name}/full_name_${name}_seed_${seed}_bs_${bs}_ss_${max_length}_ftlr_${lr}_ftwd_${wd}_gradual_unfreeze_${gradual_unfreeze}_discriminate_${discriminate}.txt
                # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch
                CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch example.py --name ${name} --weight ${weight} --lr ${lr} --wd ${wd} --seed ${seed} --bs ${bs} --max_length ${max_length} --gradual_unfreeze ${gradual_unfreeze} --discriminate ${discriminate} --use_smaller_vocab ${use_smaller_vocab} 2>&1 | tee ${log_file}
                end=`date +%s`
                runtime=$((end-start))
                echo "time taken: ${runtime}"
            done
        done
    done
done