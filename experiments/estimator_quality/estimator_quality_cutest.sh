#!/bin/bash

ESTIMATORS=('ffd' 'cfd') # 'opt_ffd' 'opt_cfd')

for grad_estimator in "${ESTIMATORS[@]}"; do
    # base_dir d  L mu 
    nohup python3 estimator_quality_cutest.py ${grad_estimator} ${1} 0.0 10 &
done