# !/bin/bash

TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TRAINING_LOG=test_38_${TRAINING_TIMESTAMP}.log
echo "$TRAINING_LOG"
# mkdir log
python test_metric_score.py 2>&1 | tee ./log/$TRAINING_LOG
