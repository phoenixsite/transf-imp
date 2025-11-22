#!/bin/bash

#set -e

LOG_LEVEL=30
BATCH_SIZE=10
PROGRAM=run_transferability.py

RESULT_DIR=../result_transfer_nformula_pgd

for test_dir in ../result-nformula/2025-11-03/*
do
        aes_dir=$(find "$test_dir" -name "adversarial_examples" -type d)
        echo "Using $aes_dir directory"
        for target_model in "neural-representation-purifier" "resnet152.tv2_in1k"
        do
                echo "Testing model: $target_model"
                uv run "$PROGRAM" \
                        -d "$aes_dir" \
                        -tm "$target_model" \
                        --log-level "$LOG_LEVEL" \
                        -o "$RESULT_DIR/$target_model" \
                        -g 0 \
                        --test-transferability \
                        --test-samples \
                        -bs "$BATCH_SIZE"
        done
done

RESULT_DIR=../result_transfer_n100_pgd

for test_dir in ../result-n100/2025-11-03/*
do
        aes_dir=$(find "$test_dir" -name "adversarial_examples" -type d)
        echo "Using $aes_dir directory"
        for target_model in "neural-representation-purifier" "resnet152.tv2_in1k"
        do
                echo "Testing model: $target_model"
                uv run "$PROGRAM" \
                        -d "$aes_dir" \
                        -tm "$target_model" \
                        --log-level "$LOG_LEVEL" \
                        -o "$RESULT_DIR/$target_model" \
                        -g 0 \
                        --test-transferability \
                        --test-samples \
                        -bs "$BATCH_SIZE"
        done
done
