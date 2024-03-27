# !/bin/bash

TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TRAINING_LOG=${TRAINING_TIMESTAMP}.log
echo "$TRAINING_LOG"
# mkdir log
bash run.sh $1 2>&1 | tee ./log/$TRAINING_LOG