#!/bin/bash

ESTIMATORS=('ffd' 'cfd')


for grad_estimator in "${ESTIMATORS[@]}"; do
    nohup python3 estimator_quality_ls.py ${grad_estimator} ${1} ${2} 1e-1 0.0 ${3} ${4} 100 &
    nohup python3 estimator_quality_ls.py ${grad_estimator} ${1} ${2} 1e-3 0.0 ${3} ${4} 100 &
    nohup python3 estimator_quality_ls.py ${grad_estimator} ${1} ${2} 1e-5 0.0 ${3} ${4} 100 &
    nohup python3 estimator_quality_ls.py ${grad_estimator} ${1} ${2} 1e-7 0.0 ${3} ${4} 100 &
done