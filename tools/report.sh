#!/bin/bash
TABLE="|config|train|valid|test|wps|time|"
TABLE="$TABLE\n|---|---|---|---|---|---|"

commit=$(git rev-parse HEAD)
comment=$(git log --oneline -1 --pretty=%B)
user=$(git config user.name)
date=$(date --rfc-3339='date')

result="$user_$date"

# Finding the right filename (format user_date.md or user_date.1.md .2.md if multiple result the same day
i=0
filename(){
  if [ $i == 0 ]; then
    echo "./results/${user}_${date}.md"
  else
    echo "./results/${user}_${date}.${i}.md"
  fi

}
while [ -e "$(filename)" ]; do
  i=$(($i+1))
done


# Asking some details
printf "Which dataset did you used? (single line description)\n"
read -e DATASET

printf "Which hardware was involved? (single line description)\n"
read -e HARDWARE

if [ ${#DATASET} == 0 ]; then
  DATASET="n/a"
fi
if [ ${#HARDWARE} == 0 ]; then
  HARDWARE="n/a"
fi

userlink(){
  echo "[$1](https://github.com/$1)"
}

commitlink(){
  commit="$1"
  shortcommit=$(echo $commit | cut -c-7)
  comment="$2"
  
  echo "[$comment $shortcommit](https://github.com/pltrdy/tf_rnnlm/commit/$1)"  
}

mean(){
  sum=0
  count=0
  for i in $(echo $1 | cut -f2 -d-)
  do
    sum=$(($sum+$i))
    count=$(($count+1))
  done
  echo $(($sum/$count))
}

readconf(){
  conf="$1"
  printf "\n\`\`\`json\n"
  cat "$conf/config"
  printf "\n\`\`\`\n"
}

reportconf(){
  conf="$1"
  printf "\n## $conf\n"
  if [ -d $1 ]; then
    readconf "$1"
    train=$(tail $conf/train.output | grep "Train" | awk '{print $NF}')
    valid=$(tail $conf/train.output | grep "Valid" | awk '{print $NF}')
    real=$(tail $conf/train.output | grep "real" | awk '{print $NF}')
    tst=$(tail $conf/test.output | grep "Test" | awk '{print $NF}')  
    wps=$(cat $conf/train.output | grep "wps" | awk '{print $(NF-1)}')
    meanwps=$(mean "$wps")

    #list "Training PPL" "$train"
    #list "Valid PPL" "$valid"
    #list "Speed" "$meanwps wps"
    #list "Time" "$real"

    TABLE="$TABLE\n|$1|$train|$valid|$tst|$meanwps|$real|"
  else
    printf "No '$1'\n"
  fi
}

list(){
  printf " * **$1**: $2\n"
}

generate(){
    
  echo "## About"
  list "User" "$(userlink $user)"
  list "Date" "$date"
  list "Commit" "$(commitlink "$commit" "$comment")"
  list "Dataset" "$DATASET"
  list "Hardware" "$HARDWARE"


  reportconf "small"
  reportconf "medium"
  reportconf "large"
  printf "\n## Results"
  printf "\n$TABLE"
}

git diff --exit-code &> /dev/null
ret="$?"
if [ $ret == 1 ]; then
  echo "Please commit your changes before generating a report"
  exit 1
fi

echo "Using username: $user"
echo "Using commit: $comment $commit"
generate > $(filename)
echo "Report generated: $(filename)"
echo "Please fill missing informations:"
cat -n $(filename) | grep "n/a"
