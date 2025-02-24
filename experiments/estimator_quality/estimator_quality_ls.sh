#!/bin/bash

ESTIMATORS=('ffd' 'cfd')


for grad_estimator in "${ESTIMATORS[@]}"; do
    nohup python3 estimator_quality_ls.py ${grad_estimator} ${1} ${2} 1e-1 0.0 1 ${3} ${4} &
done