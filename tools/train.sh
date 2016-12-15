#!/bin/bash

# This is an example showing how we can use a model
# trained using sampled softmax to test with optimal
# performances.
# The trick is to use a model with transposed weights

MODEL_DIR='./model'
DATA_PATH='./data'
LOG=5
LOSS='sampledsoftmax'
CONFIG='smlg'

# Generating the "new" model (with transposed weights)
mkdir -p $MODEL_DIR

time python word_lm.py --action train --data_path $DATA_PATH --model_dir $MODEL_DIR --config $CONFIG --loss $LOSS --log_rate $LOG
