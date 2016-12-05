#!/bin/bash

# This is an example showing how we can use a model
# trained using sampled softmax to test with optimal
# performances.
# The trick is to use a model with transposed weights

MODEL_PATH='./model'
TR_PATH='./model_tr'
DATA_PATH='./data'

echo "This script will generate a new model using transposed weights for improved perfomances"
echo "Reading from: $MODEL_PATH"
echo "Writing to: $TR_PATH"
read -p "Are you sure? [Yy] " -n 1 -r
echo    
if [[ ! $REPLY =~ ^[Yy]$ ]]
then

  # Generating the "new" model (with transposed weights)
  mkdir -p $TR_PATH
  echo "===== Transpose ====="
  time python transpose.py --src $MODEL_PATH --dst $TR_PATH
  echo "=====  ====="

  # Testing it
  echo "===== Test ====="
  time python word_lm.py --action test --data_path $DATA_PATH --model_dir $TR_PATH

fi
