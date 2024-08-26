from configuration import BaseConfig
import argparse
import os
from configuration import ExternalEvaluationConfig
from datahandler import DataReader, DataWriter
from src import EvaluationMetrics
from src import InferenceDatasetFactory
from src import InferenceFactory
import pandas
from functools import reduce

def apply_answer(pred, label_list):
    predict_list = []
    for label in label_list:
        if label.lower() in pred or pred in label.lower():
            predict_list.append(label)
    
    return predict_list

def filter_empty_lists(lists):
    return [lst for lst in lists if lst]

def common_elements(lists):
    if not lists:
        return []
    return list(reduce(set.intersection, map(set, lists)))

def get_common_elements(rows):
    non_empty_lists = filter_empty_lists(rows)
    return common_elements(non_empty_lists)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_name", type=str, required=True)
    parser.add_argument("-template_all", "--template_all", nargs='+', required=True)
    parser.add_argument("-model_all", "--model_all", nargs='+', required=True)
    config = ExternalEvaluationConfig().get_args()
    
    if config.kb_name == 'medcin' or config.kb_name == 'snomedct_us' or config.kb_name == 'nci':
        datasets_dir = f'datasets/UMLS/{config.kb_name}_entities.json'
    else:
        datasets_dir = f'datasets/{config.kb_name.upper()}/{config.kb_name}_entities.json'
    
    # if config.kb_name == 'medcin' or config.kb_name == 'snomedct_us' or config.kb_name == 'nci':
    #     label = pandas.DataFrame(DataReader.load_json(datasets_dir))
    #     label = label[label.status.isin(['test'])]["type"].values.tolist() 
    # else:
    #     label = pandas.DataFrame(DataReader.load_json(datasets_dir)["test"])["label"]
    labels = []
    count = 0
    for template in config.template_all:
        print(template)
        outputs_pred = pandas.DataFrame()
        
        for model in config.model_all:
            output_dir = os.path.join(config.root_dir, config.kb_name, model)
            
            if not os.path.exists(output_dir):
                print(f"'{output_dir}' is not existed!")
                exit(0)

            reports_file, outputs_file = [], []
            for file in os.listdir(output_dir):
                if 'report' in file and template in file:
                    reports_file.append(file)
                elif "output" in file and template in file:
                    outputs_file.append(file)
            
            assert len(reports_file) == 1
            assert len(outputs_file) != 0

            output_file_path = os.path.join(output_dir, outputs_file[0])
            outputs_model = pandas.DataFrame(DataReader.load_json(output_file_path)['outputs'])

            if count < 1:
                for output in DataReader.load_json(output_file_path)['outputs']:
                    label_list = output['label']
                    labels.append(label_list)
                count += 1

            if "bloom" in model or "flan" in model or "llama" in model:
                outputs_pred[model] = outputs_model.apply(lambda row: apply_answer(row.pred[0].lower().rstrip('\n').strip(), row.label) , axis = 1)
            else:
                outputs_pred[model] = outputs_model["pred"].apply(lambda x: x[0])

        # most_common_elements = outputs_pred.apply(lambda x: x.mode().iloc[0], axis=1)
        most_common_elements = outputs_pred.apply(lambda x: get_common_elements(x), axis=1)
        outputs_pred["final_result"] = most_common_elements
        outputs_pred["label"] = labels
        outputs_pred.to_csv(f'{template}.csv', index=False)

        evaluator = EvaluationMetrics(ks=config.eval_ks, metric=config.eval_metric)
        results = evaluator.evaluate(actual=labels, predicted=outputs_pred["final_result"])
        print("Results:", results)
