#!/bin/bash

set -e

PROGRAM=run_imperceptibility.py
BS=300

dirs=(results/inception* results/resnet* results/vgg*)

input=()
for d in "${dirs[@]}"; do
    [[ -e "$d" ]] && input+=("$d")
done

uv run "$PROGRAM" \
    --nthreads 10  \
    --batch-size $BS \
    --gpu \
    --append \
    "similarity_fid.csv" \
    "${input[@]}"