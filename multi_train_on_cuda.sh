#!/usr/bin/env bash

set -e
LOG_DIR=logs
mkdir -p ${LOG_DIR}
mkdir -p train_model

DEFAULT_GPU_NUM=8
if [ -z "$1" ]; then
    GPU_NUM=${DEFAULT_GPU_NUM}
    echo "[INFO] GPU_NUM not specified, using default: ${GPU_NUM}"
else
    GPU_NUM=$1
fi

# check if GPU_NUM is a number
if ! [[ "${GPU_NUM}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_NUM must be a number."
    echo "Usage: $0 [GPU_NUM]"
    echo "Example:"
    echo "  $0            # use default GPU_NUM=${DEFAULT_GPU_NUM}"
    echo "  $0 8          # use 8 GPUs"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE=${LOG_DIR}/train_${TIMESTAMP}_${GPU_NUM}_gpus.log

echo "[INFO] Using GPU_NUM=${GPU_NUM}"
echo "[INFO] Logging to ${LOG_FILE}"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# for debug
# export TORCH_SHOW_CPP_STACKTRACES=1
# export TORCH_CPP_LOG_LEVEL=INFO


torchrun --nproc_per_node=${GPU_NUM} \
    run.py ../datasets/RetailRocket/retailrocket_processed_view_train_full.tsv \
    -t ../datasets/RetailRocket/retailrocket_processed_view_test.tsv \
    -m 1 5 10 20 \
    -ps layers=224,batch_size=80,dropout_p_embed=0.5,dropout_p_hidden=0.05,learning_rate=0.05,momentum=0.4,n_sample=2048,sample_alpha=0.4,bpreg=1.95,logq=0.0,loss=bpr-max,constrained_embedding=True,elu_param=0.5,n_epochs=10 \
    -s ./train_model/save_model.pt \
    2>&1 | tee ${LOG_FILE}

#后台运行参考
# nohup bash train_on_musa.sh > /dev/null 2>&1 &