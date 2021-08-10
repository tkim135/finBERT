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

start=`date +%s`
log_file="gpt_downstream/test.txt"
CUDA_VISIBLE_DEVICES=0 python3 -u run_seeds.py --model_name_or_path "gpt2" 2>&1 --start_seed 0 --end_seed 2 --type public | tee ${log_file}
end=`date +%s`
runtime=$((end-start))
echo "time taken: ${runtime}"
rm -rf ~/finbert_clpath/public/