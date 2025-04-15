#!/bin/bash

ESTIMATORS=('ffd') # 'cfd' 'opt_ffd' 'opt_cfd')
DIMENSIONS=(100 500 1000) 

for d in "${DIMENSIONS[@]}"; do
for grad_estimator in "${ESTIMATORS[@]}"; do
    # base_dir  L mu 
    nohup python3 estimator_quality.py 'logistic' ${grad_estimator} ${d} --sigma 0.0 --n 1000 --lam 1e-5 --out_dir ${1} --reps 50 --num_init 10 --h 1e-7 &
done
done

wait