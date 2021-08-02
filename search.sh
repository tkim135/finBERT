#!/bin/bash

for file in /scratch/varunt/finBERT/downstream_eval_gridsearch_results/public/* ; do
    wc -l ${file}
done