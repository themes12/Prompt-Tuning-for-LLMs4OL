#!/bin/bash

label="prompt-tuning"
device="cuda"
# datasets=("wn18rr" "geonames" "nci" "snomedct_us" "medcin")
datasets=("molecular")
#templates=("template-1" "template-2" "template-3" "template-4" "template-5" "template-6" "template-7" "template-8")
templates=("template-3")
#templates=("template-8")
#models=("bloom_1b7" "bloom_3b" "llama_7b" )
models=("llama_7b" "llama2" "llama2_chat" "llama3" "llama3_chat")
#models=("bloom_1b7")
train_size="0.05"
soft_prompt="PT"
virtual_token="30"

for kb_name in "${datasets[@]}"; do
  index=1
  for model_name in  "${models[@]}"; do
    log="results/$kb_name/$index-$kb_name-$model_name.$label.test.log.txt"
    exec > $log
    for template in "${templates[@]}"; do  
      #for train_size in "${train_sizes[@]}"; do
      echo "Running on dataset: $kb_name , model: $model_name, template: $template, soft prompt: $soft_prompt , virtual token: $virtual_token, train size: $train_size!"
      CUDA_VISIBLE_DEVICES=1 python3 prepare_prompt_tuning.py --kb_name=$kb_name --model_name=$model_name --template=$template --device=$device --soft_prompt=$soft_prompt --virtual_token=$virtual_token --train_size=$train_size
      #python3 test.py --kb_name=$kb_name --model_name=$model_name --template=$template --device=$device
      echo "Prompt training for $model_name on template: $template is done"
      echo "-----------------------------------------------------------"
      #done
    done
    index=$((index+1))
  done
done


# datasets=("wn18rr" "nci" "snomedct_us" "medcin")
: 'datasets=("wn18rr")
templates=("template-1" "template-2" "template-3" "template-4" "template-5" "template-6" "template-7" "template-8")
models=("gpt3" "chatgpt")
for kb_name in "${datasets[@]}"; do
  index=8
  for model_name in  "${models[@]}"; do
    log="results/$kb_name/$index-$kb_name-$model_name.$label.test.log.txt"
    exec > $log
    for template in "${templates[@]}"; do
      echo "Running on dataset: $kb_name , model: $model_name, template: $template!"
      python3 test.py --kb_name=$kb_name --model_name=$model_name --template=$template --device=$device
      echo "Inference for  $model_name on template: $template  is done"
      echo "-----------------------------------------------------------"
    done
    index=$((index+1))
  done
done
'

: 'datasets=("geonames")
templates=("template-1")
models=("gpt3" "chatgpt")
for kb_name in "${datasets[@]}"; do
  index=10
  for model_name in  "${models[@]}"; do
    log="results/$kb_name/$index-$kb_name-$model_name.$label.test.log.txt"
    exec > $log
    for template in "${templates[@]}"; do
      echo "Running on dataset: $kb_name , model: $model_name, template: $template!"
      python3 test.py --kb_name=$kb_name --model_name=$model_name --template=$template --device=$device
      echo "Inference for  $model_name on template: $template  is done"
      echo "-----------------------------------------------------------"
    done
    index=$((index+1))
  done
done
'

: ' datasets=("wn18rr" "geonames" "nci" "snomedct_us" "medcin")
templates=("template-3")
models=("gpt4")
for kb_name in "${datasets[@]}"; do
  index=12
  for model_name in  "${models[@]}"; do
    log="results/$kb_name/$index-$kb_name-$model_name.$label.test.log.txt"
    exec > $log
    for template in "${templates[@]}"; do
      echo "Running on dataset: $kb_name , model: $model_name, template: $template!"
      python3 test.py --kb_name=$kb_name --model_name=$model_name --template=$template --device=$device
      echo "Inference for  $model_name on template: $template  is done"
      echo "-----------------------------------------------------------"
    done
    index=$((index+1))
  done
done
'
