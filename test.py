from configuration import BaseConfig
from datahandler import DataReader, DataWriter
from src import InferenceDatasetFactory
from src import InferenceFactory
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from src import EvaluationMetrics
import datetime
import os
# import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#openai.api_key  = os.environ['OPENAI_API_KEY']

def inference(model, dataloader):
    # print("predicting")
    predictions, logits, labels, batches, ids = [], [], [], [], []
    for batch in tqdm(dataloader):
        prediction, logit = model.make_batch_prediction(list(batch['sample']))
        #break
        for pred, log, label, batch_sample, id in zip(prediction, logit, list(batch['label']), list(batch['sample']), list(batch['ID'])):
            predictions.append(pred)
            logits.append(list(log))
            labels.append(label)
            batches.append(batch_sample)
            ids.append(id)
    return predictions, logits, labels, batches, ids

def gpt_inference(model, dataloader, config):
    for index, batch in enumerate(tqdm(dataloader)):
        results = model.make_batch_prediction(batch)
        DataWriter.write_json(results,  config.model_output.replace("[BATCH]", str(index+1)))
        print(f"scoring model outputs in:{config.model_output}")
                                  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--soft_prompt", type=str, choices=["PT", "PRET", "P-T", "MPT"], default=None)
    parser.add_argument("--virtual_token", type=int, default=None)
    parser.add_argument("--train_size", type=float, default=None)
    args = parser.parse_args()
   
    print("args:", args)
    config = BaseConfig(version=3).get_args(kb_name=args.kb_name, model=args.model_name, template=args.template, device=args.device, soft_prompt=args.soft_prompt)
    start_time = datetime.datetime.now()
    print("Starting the Inference time is:", str(start_time).split('.')[0])
    dataset = DataReader.load_json(config.entity_path)
    print(config.entity_path)
    if args.kb_name != "cellular":
        templates = DataReader.load_json(config.templates_json)[config.template_name]
    else:
        templates = DataReader.load_json(f'./datasets/GO/templates_cellular.json')[config.template_name]

    if args.kb_name == "geonames":
        dataset = dataset["geonames"]
    elif args.kb_name == "wn18rr":
        dataset = dataset["test"]

    try:
        label_mapper = DataReader.load_json(config.label_mapper)
    except:
        label_mapper = None

    test_dataset = InferenceDatasetFactory(kb_name=args.kb_name, data=dataset, templates=templates,
                                           template=args.template, label_mapper=label_mapper)
    
    print(test_dataset)

    report_dict = {
        "baseline-run-args": str(args),
        "report_output_refrence": config.report_output,
        "results":[],
        "dataset-in-use": str(test_dataset),
        "configs": vars(config)
    }
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    inference_model = InferenceFactory(config)(model_name=args.model_name)
    
    if "gpt" in args.model_name:
        #  == "gpt3" or args.model_name == "gpt4"
        gpt_inference(model=inference_model, dataloader=test_dataloader, config=config)
        print(f"scoring results in:{config.report_output}")
        DataWriter.write_json(report_dict,  config.report_output)
    else:
        predictions, logits, labels, batches, ids = inference(model=inference_model, dataloader=test_dataloader)
        outputs = {
            "model-name": args.model_name,
            "dataset": report_dict['dataset-in-use'],
            "outputs": [{"ID":id, "pred":pred, "label":label} for pred, label, batch, id in zip(predictions, labels, batches, ids)]
        }
        DataWriter.write_json(outputs,  config.model_output)
        print(f"scoring model outputs in:{config.model_output}")

        evaluator = EvaluationMetrics(ks=config.eval_ks, metric=config.eval_metric)
        results = evaluator.evaluate(actual=labels, predicted=predictions)
        report_dict['results'] = results
        print("Results:", results)
        print(f"scoring results in:{config.report_output}")
        DataWriter.write_json(report_dict,  config.report_output)
    end_time = datetime.datetime.now()
    print("Ending the Inference time is:", str(end_time).split('.')[0])
    print("Total duration is===>", str(end_time - start_time))