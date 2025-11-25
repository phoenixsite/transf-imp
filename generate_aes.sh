#!/bin/bash

for eps in "4" "8" "16" "32"
do
    good_eps=$(echo "scale=9; $eps / 255" | bc)
    a=$(( eps + 4 ))
    b=$(echo "$eps * 1.25" | bc)
    b=${b%.*}
    form=$(( a < b ? a : b ))
    for niter in 100 $form
    do
        for algorithm in "PGD" "APGD" "ACG" "ReACG"
        do
            for model in "inception" "resnet50" "vgg16"
            do
                uv run run_evaluation_pgd.py \
                --cmd-param attacker_name:str:${algorithm} max_iter:int:${niter} eps:float:0${good_eps} \
                --nthreads 4 \
                --save-adversarial \
                --gpu \
                configs/${model}.yaml \
                "results/${model}-${algorithm}-${eps}-${niter}"
            done
        done
    done
done