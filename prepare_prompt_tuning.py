from configuration import BaseConfig
from datahandler import DataReader, DataWriter
from src import InferenceDatasetFactory
from src import InferenceFactory
from src import PromptTuning
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from src import EvaluationMetrics
import datetime
import os
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from huggingface_hub import login

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--soft_prompt", type=str, choices=["PT", "PRET", "P-T", "MPT"], default=None)
    parser.add_argument("--virtual_token", type=int, required=True)
    parser.add_argument("--train_size", type=float, required=True)
    args = parser.parse_args()
  
    print("args:", args)
    config = BaseConfig(version=3).get_args(kb_name=args.kb_name, model=args.model_name, template=args.template, device=args.device, soft_prompt=args.soft_prompt)
    start_time = datetime.datetime.now()
    print("Starting the Inference time is:", str(start_time).split('.')[0])
    # dataset = DataReader.load_json(config.entity_path)
    dataset = DataReader.load_json(config.entity_path)
    templates = DataReader.load_json(config.templates_json)[config.template_name]

    try:
        label_mapper = DataReader.load_json(config.label_mapper)
    except:
        label_mapper = None

    if args.kb_name == "wn18rr":
        dataset = pd.json_normalize(dataset["train"]) # wn18rr
    elif args.kb_name == "geonames":
        dataset = pd.json_normalize(dataset["geonames"]) # other
        dataset = dataset[dataset["status"] == 'train'] # other
        dataset = dataset.sample(n=50000, random_state=42)
    else:
        dataset = pd.json_normalize(dataset) # other
        dataset = dataset[dataset["status"] == 'train'] # other
        dataset = dataset.sample(n=50000, random_state=42)
    
    dataset = Dataset.from_pandas(dataset)
    dataset = Dataset.from_list(InferenceDatasetFactory(kb_name=args.kb_name, data=dataset, templates=templates, template=args.template, label_mapper=label_mapper, prompt_tune=True, is_train=True))

    dataset = dataset.train_test_split(train_size=args.train_size, seed=42)

    model = PromptTuning(config)(model_name=args.model_name, dataset=dataset)
    
    end_time = datetime.datetime.now()
    print("Ending the Inference time is:", str(end_time).split('.')[0])
    print("Total duration is===>", str(end_time - start_time))