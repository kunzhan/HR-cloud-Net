# !/bin/bash

# DS=38
# DS=CH
# DS=spars
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TRAINING_LOG=${TRAINING_TIMESTAMP}.log
echo "$TRAINING_LOG"
# mkdir log
bash LC08.sh 0 2>&1 | tee ./log/$TRAINING_LOG
