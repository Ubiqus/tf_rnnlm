#!/bin/bash

DATA_PATH="./simple-examples/data"
MODEL_ROOT="./ptb_models"
LOSS="sampledsoftmax"
LOG=5

TIME='time'
UNBUFFER='unbuffer'

#
# Array of different value to run
# (here --config)
#

declare -a confs=("small" "medium" "large")

#
# Useful wrapper for printing & time
# 
wrapper() {
  CONF="$1"
  CMD="$(train $CONF)"
  OUTPUT="$MODEL_ROOT/$CONF.output"
  ret=" { $TIME $UNBUFFER $CMD ; } 2>&1 | tee $OUTPUT }"	
  echo $ret
}

#
# Run the training with the parameter given
#
train() {
  data_path="--data_path $DATA_PATH"
  model_dir="--model_dir $MODEL_ROOT/$1"
  config="--config $1" 
  loss="--loss $LOSS"
  log="--log $LOG"

  echo " python word_lm.py --action train $data_path $model_dir $config $loss $log"
}

echo "Training PTB Models using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mkdir -p MODEL_ROOT

for i in "${confs[@]}"
do
   echo "\n\n===== $i =====\n\n"
   mkdir -p "$MODEL_ROOT/$i"
   eval $(wrapper "$i")  
done

#
# Match the parameter string in output files
# and print results in a file
#
printmetric(){
  metric="$1"
  files=$(ls  $MODEL_ROOT | sort -n | grep output)
  tail -n +1 $files | grep -E "$metric|=>" | grep -E [[:digit:]] | tee "$MODEL_ROOT/$metric"

}

printmetric Valid
printmetric wps

