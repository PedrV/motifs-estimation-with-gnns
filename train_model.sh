#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ -z "$1" ]; then
  echo "Usage: $0 mpgnn_id"
  exit 1
fi

GIN="" # 0
GCN="" # 1
GAT="" # 2
SAGE="" # 3

MPGNN="$1"
DECODER=0

CONTINUE_OPT=0
CONTINUE_RUN=0

LOG_SCRIPT_NAME=$SCRIPT_DIR"/ray_logs.log"
TRAIN_SCRIPT_NAME=$SCRIPT_DIR"/hephaestus/training/train_model.py"

echo "Starting train MPGNN ${MPGNN}, DECODER ${DECODER} ..."
sleep 10s 

if [[ CONTINUE_OPT -eq 1 ]]; then
    /path/to/env $SCRIPT_NAME -u --mpgnn $MPGNN --decoder $DECODER --restore-experiment $GIN --continue-optimization > $LOG_SCRIPT_NAME 2>&1
elif [[ CONTINUE_RUN -eq 1 ]]; then
    /path/to/env $SCRIPT_NAME -u --mpgnn $MPGNN --decoder $DECODER --restore-experiment $GIN --unfinished-trials > $LOG_SCRIPT_NAME 2>&1
else
    /path/to/env $SCRIPT_NAME -u --mpgnn $MPGNN --decoder $DECODER > $LOG_SCRIPT_NAME 2>&1
fi
