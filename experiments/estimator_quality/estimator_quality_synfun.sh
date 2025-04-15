#!/bin/bash

ESTIMATORS=('ffd') # 'cfd' 'opt_ffd' 'opt_cfd')
DIMENSIONS=(100 500 1000) 

for d in "${DIMENSIONS[@]}"; do
for grad_estimator in "${ESTIMATORS[@]}"; do
    # base_dir  L mu 
    nohup python3 estimator_quality.py ${1} ${grad_estimator} ${d} --sigma 0.0  --out_dir ${2} --reps 50 --num_init 10 --h 1e-7 &
done
done

wait