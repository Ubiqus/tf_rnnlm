#!/bin/bash

DATA_PATH="./simple-examples/data"
MODEL_ROOT="./batchsize_tuning"
LOSS="sampledsoftmax"
LOG=5
BASE_CONFIG="small"


TIME='time'
UNBUFFER='unbuffer'

PARAMS="--max_max_epoch 1 --config $BASE_CONFIG"


#
# Array of different config value
# (here --batch_size)
#
declare -a confs=("10" "20" "50" "100" "200")

#
# Useful wrapper for printing & time
# 
wrapper() {
  CONF="$1"
  CMD="$(train $CONF)"
  OUTPUT="$MODEL_ROOT/$CONF.tr.output"
  ret=" { $TIME $UNBUFFER $CMD ; } 2>&1 | tee $OUTPUT }"	
  echo $ret
}

#
# Run the training with the parameter given
#
train() {
  data_path="--data_path $DATA_PATH"
  model_dir="--model_dir $MODEL_ROOT/$1"
  config="--batch_size $1" 
  loss="--loss $LOSS"
  log="--log $LOG"

  echo " python word_lm.py --action train $data_path $model_dir $config $loss $log $PARAMS"
}

echo "Tuning PTB Models using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mkdir -p MODEL_ROOT

for i in "${confs[@]}"
do
   echo "===== $i ====="
   mkdir -p "$MODEL_ROOT/$i"
   eval $(wrapper "$i")  
done

#
# Match the parameter string in output files
# and print results in a file
#
printmetric(){
  metric="$1"


  # Get sorted list of file (oldest first)
  files=$(ls -rt  | grep output)
  tail -n +1 $files | egrep "$metric|=>" | egrep "[[:digit:]]" | tee "$metric"

}

dir=$(pwd)
cd "$MODEL_ROOT"

printmetric Valid
printmetric wps


