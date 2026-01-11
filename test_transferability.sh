#!/bin/bash

#set -e

BATCH_SIZE=100
PROGRAM=run_transferability.py

RESULT_DIR=./results_transfer/

dirs=(results/inception* results/resnet* results/vgg*)

target_models=("random-padding" "jpeg" "bit-reduction" "neural-representation-purifier" "vgg19.tv_in1k" "resnet152.tv2_in1k" "mobilenetv2_140.ra_in1k")

uv run "$PROGRAM" \
        --batch-size "$BATCH_SIZE" \
        --gpu \
        --nthreads 10 \
        --datadirs "${dirs[@]}" \
        --rootoutputdir "${RESULT_DIR}" \
        -- "${target_models[@]}"

# "vgg16.tv_in1k" "inception_v3.tv_in1k" "resnet50.tv2_in1k" "inception_v3.tf_adv_in1k" "inception_resnet_v2.tf_ens_adv_in1k" 

# for test_dir in ../result-nformula/2025-11-03/*
# do
#         aes_dir=$(find "$test_dir" -name "adversarial_examples" -type d)
#         echo "Using $aes_dir directory"
#         for target_model in "neural-representation-purifier" "resnet152.tv2_in1k"
#         do
#                 echo "Testing model: $target_model"
                
#         done
# done

# RESULT_DIR=../result_transfer_n100_pgd

# for test_dir in ../result-n100/2025-11-03/*
# do
#         aes_dir=$(find "$test_dir" -name "adversarial_examples" -type d)
#         echo "Using $aes_dir directory"
#         for target_model in "neural-representation-purifier" "resnet152.tv2_in1k"
#         do
#                 echo "Testing model: $target_model"
#                 uv run "$PROGRAM" \
#                         -d "$aes_dir" \
#                         -tm "$target_model" \
#                         --log-level "$LOG_LEVEL" \
#                         -o "$RESULT_DIR/$target_model" \
#                         -g 0 \
#                         --test-transferability \
#                         --test-samples \
#                         -bs "$BATCH_SIZE"
#         done
# done
