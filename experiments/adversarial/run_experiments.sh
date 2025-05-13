#!/bin/bash


out_path=$1
reps=$2

methods=("spherical" "gaussian" "rademacher" "coordinate" "householder" "butterfly" "permuted_householder" "qr")

for method in "${methods[@]}"
do
    echo "[--] Running Adversarial experiment for $method directions ($reps repetitions) -> output path: $out_path"
    nohup python adv_exp.py $out_path $reps $method &
done