#!/bin/bash
#
# This script will compare test perplexity as well
# as running time using a small model
#
# By default it will use ./batchsize_tuning/20/transposed
# which may has been created if you previosuly ran
# tools/tune_batchsize_tr.sh
#

DATA_PATH="./simple-examples/data"
MODEL_ROOT="./batchsize_tuning"

# We'll use a model with 20 batch_size
# it correspond to default 'small' model
MODEL="small"
MODEL_TRANSPOSED="$MODEL_ROOT/$MODEL/transposed"

TIME='time'
UNBUFFER='unbuffer'

PARAMS="--num_steps 10"

#
# Array of different config value
# (here --batch_size)
#
declare -a confs=("1" "2" "5" "10" "20" "50" "100" "200" "500")

#
# Useful wrapper for printing & time
# 
wrapper() {
  CONF="$1"
  CMD="$(tst $CONF)"
  OUTPUT="$MODEL_ROOT/$CONF.tst.output"
  ret=" { $TIME $UNBUFFER $CMD ; } 2>&1 | tee $OUTPUT }"	
  echo $ret
}

#
# Run the training with the parameter given
#
tst() {
  data_path="--data_path $DATA_PATH"
  model_dir="--model_dir $MODEL_TRANSPOSED"
  config="--batch_size $1" 

  echo " python word_lm.py --action test $data_path $model_dir $config $PARAMS"
}

echo "Tuning PTB Models using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ ! -d "$MODEL_TRANSPOSED" ]; then
  echo "No transposed model"
  echo "Creating: $MODEL_TRANSPOSED"
  ./transpose.py --src "$MODEL_ROOT/$MODEL" --dst "$MODEL_TRANSPOSED" &> /dev/null
  echo "Done."
else
  echo "Using transposed model: $MODEL_TRANSPOSED"
fi


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
  files=$(ls -rt | grep tst.output)
  tail -n +1 $files | egrep "$metric|=>" | egrep "[[:digit:]]" | tee "$metric.tst"

}

dir=$(pwd)
cd "$MODEL_ROOT"

printmetric Perplexity
printmetric user

cd "$dir"


