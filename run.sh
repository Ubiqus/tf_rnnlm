#!/bin/bash

# Dec. 2016 - pltrdy
# Train&Test small, medium and large model
# and generate a report 

DATA_PATH="./data"
LOSS='sampledsoftmax'
LOG_RATE=3
BATCH_SIZE=64
NUM_SAMPLES=1024

if ! [ -d "$DATA_PATH" ]; then
  echo "No ./data directory found."
  if ! [ -d "./simple-examples" ]; then
    ./tools/get_ptb.sh
  else
    echo "Using ./simple-examples"
    echo "Creating symlink ./$DATA_PATH -> ./simple-examples/data"
    ln -s ./simple-examples/ "$DATA_PATH"
  fi
  
else
  echo "Using data directory $DATA_PATH"
fi

train(){
  conf="$1"
  model_dir="--model_dir $conf"
  data_path="--data_path $DATA_PATH"
  loss="--loss $LOSS"
  config="--config $conf"
  batch_size="--batch_size $BATCH_SIZE"
  num_samples="--num_samples $NUM_SAMPLES"
  log="--log_rate $LOG_RATE"
  output="./$conf/train.output"
  
  cmd="{ time unbuffer python word_lm.py --action train $model_dir $data_path $config $loss $batch_size $num_samples $log ;} 2>&1 | tee $output "
  echo "$cmd"
}

transpose(){
  conf="$1"

  cmd="./transpose.py --src $conf --dst $conf/transposed"
  echo "$cmd"
}

test_transposed(){
  conf="$1"
  model_dir="--model_dir $conf/transposed"
  data_path="--data_path $DATA_PATH"
  output="./$conf/test.output"
  
  cmd="{ time python word_lm.py --action test $model_dir $data_path ;} 2>&1 | tee $output "
  echo "$cmd"
}

run(){
  conf="$1"
  echo "Running configuration: $1"
  mkdir -p $1
  eval $(train $conf)
  eval $(transpose $conf)
  eval $(test_transposed $conf)

}

echo $(pwd)
run "small"
run "medium"
run "large"

printf "PTB dataset\nn/a" | ./tools/report.sh
