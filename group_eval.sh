#!/bin/bash

dataset="geonames"
templates=("template-1" "template-2" "template-3" "template-4" "template-5" "template-6" "template-7" "template-8")
#templates=("template-2" "template-3" "template-4" "template-8")
models=("bloom_1b7" "bloom_3b" "llama_7b" "llama2" "llama2_chat" "llama3" "llama3_chat")
# models=("bert_large")
python3 group_eval.py --kb_name=$dataset --template_all "${templates[@]}" --model_all "${models[@]}"