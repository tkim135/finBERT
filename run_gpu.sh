#!/bin/bash

#"""
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr1e-3_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr1e-4_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr1e-5_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr5e-5_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr5e-6_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr1e-3_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr1e-4_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr1e-5_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr5e-5_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr5e-6_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr1e-3_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr1e-4_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr1e-5_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr5e-5_ss512_bs256_results_finbert_dir
#/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr5e-6_ss512_bs256_results_finbert_dir
#"""

myArray=("/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.0005_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.0005_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.0005_lr5e-7_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.001_lr1e-4_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.001_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.001_lr1e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.001_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.001_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.001_lr5e-7_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.005_lr1e-4_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.005_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.005_lr1e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.005_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.005_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.005_lr5e-7_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.01_lr1e-4_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.01_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.01_lr1e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.01_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.01_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/hfrun_decay0.01_lr5e-7_ss512_bs256_results_finbert_dir")

#myArray=("/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr5e-6_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr5e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr1e-5_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr1e-4_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr1e-4_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr1e-4_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.01_lr1e-3_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.005_lr1e-3_ss512_bs256_results_finbert_dir" "/scratch/varunt/finBERT/models/run_ss512_bs_256/hfrun_decay0.001_lr1e-3_ss512_bs256_results_finbert_dir")

# for t in ${allThreads[@]}; do
#for folder in /scratch/varunt/finBERT/models/run_ss512_bs_256/*; do
#for folder in ${myArray[@]}; do
#for folder in /scratch/varunt/finBERT/models/run_pile_ss512_bs_256/*; do
#for folder in /scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_pile/*; do
for folder in ${myArray[@]}; do
    echo "${folder}"
    run_folder=$(echo "${folder}/" | cut -d'/' -f8)
    echo "${run_folder}"
    log_file="downstream_eval_gridsearch_results/pile/downstreamgrid${run_folder}.txt"
    echo "${log_file}"
    CUDA_VISIBLE_DEVICES=1 python3 -u run_seeds.py --model_name_or_path ${folder} 2>&1 --start_seed 0 --end_seed 4 --type pile | tee ${log_file}
    rm -rf /scratch/varunt/finbert_clpath/pile/
done

# for folder in /scratch/varunt/finBERT/models/downstream_eval_gridsearch/pretraining_grid_public/*; do
#     echo "${folder}"
#     run_folder=$(echo "${folder}/" | cut -d'/' -f7)
#     echo "${run_folder}"
#     log_file="downstream_eval_gridsearch_results/downstreamgrid_${run_folder}.txt"
#     echo "${log_file}"
#     #CUDA_VISIBLE_DEVICES=1 python3 -u run_seeds.py --model_name_or_path ${folder} 2>&1 --start_seed 0 --end_seed 4 | tee ${log_file}
#     #rm -rf /scratch/varunt/finbert_clpath/
# done