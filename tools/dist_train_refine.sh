#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500} 

#export TORCH_HOME='/apdcephfs/private_v_huaziguo/torch_home'

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_scratch.py $CONFIG --launcher pytorch ${@:3}
