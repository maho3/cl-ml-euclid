#!/bin/bash

modelnames=('true' 'msig' 'pamico' 'mamp' 'gals_nle' 'summ_nle' 'gnn_npe')
# modelnames=('pamico')
datanames=('wC50' 'wC100' 'dC50' 'dC100')

for model in "${modelnames[@]}"; do
    for data in "${datanames[@]}"; do
        echo "Running model $model on data $data"
        python ./run_mcmc_calibration.py --data $data --model $model
    done
done