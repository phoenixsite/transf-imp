#!/bin/bash

set -e

#PROGRAM=run_evaluation.py
PROGRAM=run_evaluation_pgd.py
BATCH_SIZE=50
LOG_LEVEL=30
NTHREADS=12


# RESULTS_DIR=../result-nformula/

# for config_file in ../configs/transferability-nformula/pgd*.yaml
# do
# 	echo "Executing PGD with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done

# for config_file in ../configs/transferability-nformula/acg*.yaml
# do
# 	echo "Executing ACG with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done

# for config_file in ../configs/transferability-nformula/apgd*.yaml
# do
# 	echo  "Executing APGD with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done

# for config_file in ../configs/transferability-nformula/reacg*.yaml
# do
# 	echo "Executing ReACG with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done


RESULTS_DIR=../result-n100/

for config_file in ../configs/transferability-n100/pgd*.yaml
do
	echo "Executing PGD with $config_file"
	uv run $PROGRAM -p $config_file \
		-g 0 \
		-o $RESULTS_DIR \
		--log_level $LOG_LEVEL \
		--save-adversarial \
		--n_threads $NTHREADS \
		--cmd_param batch_size:int:$BATCH_SIZE
done

# for config_file in ../configs/transferability-n100/acg*.yaml
# do
# 	echo "Executing ACG with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done

# for config_file in ../configs/transferability-n100/apgd*.yaml
# do
# 	echo  "Executing APGD with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done

# for config_file in ../configs/transferability-n100/reacg*.yaml
# do
# 	echo "Executing ReACG with $config_file"
# 	uv run $PROGRAM -p $config_file \
# 		-g 0 \
# 		-o $RESULTS_DIR \
# 		--log_level $LOG_LEVEL \
# 		--save-adversarial \
# 		--n_threads $NTHREADS \
# 		--cmd_param batch_size:int:$BATCH_SIZE
# done