from transformers import AutoTokenizer, BertForMaskedLM, \
                         BartForConditionalGeneration, \
                         T5Tokenizer, T5ForConditionalGeneration, \
                         BloomForCausalLM, BloomTokenizerFast, \
                         LlamaForCausalLM, AutoModelForCausalLM, \
                         DataCollatorForLanguageModeling, TrainingArguments, Trainer, \
                         default_data_collator, get_linear_schedule_with_warmup, default_data_collator, \
                         AutoModelForMaskedLM
import torch
from torch.utils.data import DataLoader
import openai
import time
from tqdm import tqdm
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PrefixTuningConfig, PromptEncoderConfig, PeftModel, PeftConfig
import os

class BaseLM:

    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = self.config.device
        self.top_n = self.config.top_n
        self.max_length = 256
        self.text_column = "sample"
        self.label_column = "label"

        self.prompt_tuning_getter = {
            "PT": "Prompt tuning",
            "PRET": "Prefix tuning",
            "P-T": "P-tuning",
            "MPT": "Multitask prompt tuning"
        }

        pass

    def load(self):
        pass

    def make_batch_prediction(self, Xs: list):
        pass

    def batch_tokenize(self, Xs):
        inputs = self.tokenizer(Xs, return_tensors="pt", padding=True)
        inputs.to(self.device)
        return inputs

    def single_tokenize(self, X):
        inputs = self.tokenizer(X, return_tensors="pt")
        inputs.to(self.device)
        return inputs

    def output_cleaner(self, pred, **kwargs):
        return pred

    def predict(self, X: str):
        pass

    def prepare_prompt_tuning(self, dataset):
        pass

    def preprocess_function(self, examples):
        pass

    def load_peft_model(self):
        print(self.config.dataset)
        if self.config.kb_name == "wn18rr":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="Classify part of speech of given word in the sentence to noun, verb, adjective or adverb",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
            # elif self.config.soft_prompt == "PRET":
            #     peft_config = PrefixTuningConfig(
            #         base_model_name_or_path=self.config.model_path,
            #         task_type=TaskType.CAUSAL_LM, 
            #         num_virtual_tokens=20
            #     )
            #     return get_peft_model(self.model, peft_config)
            elif self.config.soft_prompt == "P-T":
                peft_config = PromptEncoderConfig(
                    base_model_name_or_path=self.config.model_path,
                    task_type=TaskType.CAUSAL_LM,
                    num_virtual_tokens=self.config.virtual_token,
                    encoder_hidden_size=128
                )
                return get_peft_model(self.model, peft_config)
            elif self.config.soft_prompt == "MPT":
                return self.model
        elif self.config.kb_name == "geonames":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify geographical types of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        elif self.config.kb_name == "medcin":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify medical terminology of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        elif self.config.kb_name == "nci":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify medical terminology of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        elif self.config.kb_name == "snomedct_us":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify medical terminology of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        elif self.config.kb_name == "biological":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify biochemical, biomedical or biological terminology of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        elif self.config.kb_name == "cellular":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify biochemical, biomedical or biological terminology of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        elif self.config.kb_name == "molecular":
            if self.config.soft_prompt == "PT":
                peft_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=self.config.virtual_token,
                    prompt_tuning_init_text="multi class classify biochemical, biomedical or biological terminology of given word in the sentence",
                    tokenizer_name_or_path=self.config.model_path,
                )
                return get_peft_model(self.model, peft_config)
        
class MaskedLM(BaseLM):
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = BertForMaskedLM.from_pretrained(self.config.model_path)

        if self.config.soft_prompt:
            self.peft_text = self.prompt_tuning_getter.get(self.config.soft_prompt)
            self.model = PeftModel.from_pretrained(self.model, f"{self.config.peft_model_path}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")
            print(f"Loaded BertForMaskedLM from {self.config.model_path} with {self.peft_text}")
        else:
            print(f"Loaded BertForMaskedLM from {self.config.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_function(self, examples):
        inputs = []
        targets = []
        for i in range(len(examples[self.text_column])):
            sentence = examples[self.text_column][i]
            label = examples[self.label_column][i]
            sample = sentence.replace("[MASK]", label)
            inputs.append(sample)
            targets.append(str(label))

        model_inputs = self.tokenizer(inputs, padding="max_length", max_length=self.max_length)
        labels = self.tokenizer(targets, padding="max_length", max_length=self.max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_prompt_tuning(self, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = BertForMaskedLM.from_pretrained(self.config.model_path)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        self.model = self.load_peft_model()
        self.model.print_trainable_parameters()

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        training_args = TrainingArguments("trainer", evaluation_strategy="epoch", num_train_epochs=5, learning_rate=0.003)
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(f"./prompt_tuning/{self.config.kb_name}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")

    def predict(self, X:str):
        inputs = self.single_tokenize(X)
        with torch.no_grad():
            token_logits = self.model(**inputs).logits
        token_logits = token_logits.cpu()
        inputs = inputs.to('cpu')
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_n_tokens = torch.topk(mask_token_logits, self.top_n, dim=1)
        predictions, logits = [], []
        for indice, logit in zip(top_n_tokens.indices[0].tolist(), top_n_tokens.values[0].tolist()):
            predictions.append(self.output_cleaner(self.tokenizer.decode([indice])))
            logits.append(logit)
        return predictions, logits

    def make_batch_prediction(self, Xs):
        inputs = self.batch_tokenize(Xs)
        with torch.no_grad():
            token_logits = self.model(**inputs).logits
        token_logits = token_logits.cpu()
        inputs = inputs.to('cpu')
        batch_predictions, batch_logits = [], []
        for index, _ in enumerate(token_logits):
            mask_token_index = torch.where(torch.tensor([list(inputs["input_ids"][index].numpy())]) == self.tokenizer.mask_token_id)[1]
            mask_token_logits = token_logits[index, mask_token_index, :]
            top_n_tokens = torch.topk(mask_token_logits, self.top_n, dim=1)
            predictions, logits = [], []
            for indice, logit in zip(top_n_tokens.indices[0].tolist(), top_n_tokens.values[0].tolist()):
                predictions.append(self.output_cleaner(self.tokenizer.decode([indice])))
                logits.append(logit)
            batch_predictions.append(predictions)
            batch_logits.append(logits)
        return batch_predictions, batch_logits

class BARTMaskedLM(MaskedLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_path, forced_bos_token_id=0)
        
        if self.config.soft_prompt:
            self.peft_text = self.prompt_tuning_getter.get(self.config.soft_prompt)
            self.model = PeftModel.from_pretrained(self.model, f"{self.config.peft_model_path}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")
            print(f"Loaded BartForConditionalGeneration from {self.config.model_path} with {self.peft_text}")
        else:
            print(f"Loaded BartForConditionalGeneration from {self.config.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def output_cleaner(self, pred, **kwargs):
        return pred.strip()
    
    def preprocess_function(self, examples):
        inputs = []
        targets = []
        for i in range(len(examples[self.text_column])):
            sentence = examples[self.text_column][i]
            label = examples[self.label_column][i]
            sample = sentence.replace("<mask>", label)
            inputs.append(sample)
            targets.append(str(label))

        model_inputs = self.tokenizer(inputs, padding="max_length", max_length=self.max_length)
        labels = self.tokenizer(targets, padding="max_length", max_length=self.max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_prompt_tuning(self, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_path)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        self.model = self.load_peft_model()
        self.model.print_trainable_parameters()

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        training_args = TrainingArguments("trainer", evaluation_strategy="epoch", num_train_epochs=5, learning_rate=0.003)
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(f"./prompt_tuning/{self.config.kb_name}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")

class EncoderDecoderLM(BaseLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def predict(self, X: str):
        inputs = self.single_tokenize(X)
        with torch.no_grad():
            sequence_ids = self.model.generate(inputs.input_ids,
                                               num_beams=50,
                                               num_return_sequences=self.top_n,
                                               max_length=5)
        sequences = self.tokenizer.batch_decode(sequence_ids, skip_special_tokens=True)
        sequences = [self.output_cleaner(seq, prompt=X) for seq in sequences]
        logits = [0 for seq in sequences]
        return sequences, logits

    def make_batch_prediction(self, Xs):
        predictions, logits = [], []
        inputs = self.batch_tokenize(Xs)
        with torch.no_grad():
            # if self.config.soft_prompt:
                # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            sequence_ids = self.model.generate(input_ids=inputs["input_ids"],
                                               attention_mask=inputs["attention_mask"],
                                               max_new_tokens=5)
        sequences = self.tokenizer.batch_decode(sequence_ids.cpu(), skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        sequences_logist = [0 for _ in sequences]
        for index in range(0, len(Xs)):
            predictions.append(sequences[self.top_n * index:self.top_n * (index + 1)])
            logits.append(sequences_logist[self.top_n * index:self.top_n * (index + 1)])
        predictions = [[self.output_cleaner(predict, prompt=prompt) for predict in predicts]
                       for predicts, prompt in zip(predictions, Xs) ]
        return predictions, logits

    def make_single_batch_prediction(self, Xs):
        predictions, logits = [], []
        for X in Xs:
            predict, logit = self.predict(X)
            predictions.append(predict)
            logits.append(logit)
        return predictions, logits

class FlanT5EncoderDecoderLM(EncoderDecoderLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_path, device_map="balanced")

        if self.config.soft_prompt:
            self.peft_text = self.prompt_tuning_getter.get(self.config.soft_prompt)
            self.model = PeftModel.from_pretrained(self.model, f"{self.config.peft_model_path}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")
            print(f"Loaded T5ForConditionalGeneration from {self.config.model_path} with {self.peft_text}")
        else:
            print(f"Loaded T5ForConditionalGeneration from {self.config.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def batch_tokenize(self, Xs):
        inputs = self.tokenizer(Xs,
                                return_tensors="pt",
                                truncation=True,
                                padding='max_length',
                                max_length=256)
        # inputs = self.tokenizer(Xs,return_tensors="pt")
        inputs.to(self.device)
        return inputs

    def output_cleaner(self, pred, **kwargs):
        return pred.replace("<pad>", "").replace("</s>", "").strip()
    
    def preprocess_function(self, examples):
        inputs = []
        targets = []
        for i in range(len(examples[self.text_column])):
            sentence = examples[self.text_column][i]
            label = examples[self.label_column][i]
            sample = sentence.replace("?", label)
            inputs.append(sample)
            targets.append(str(label))

        model_inputs = self.tokenizer(inputs, padding="max_length", max_length=self.max_length)
        labels = self.tokenizer(targets, padding="max_length", max_length=self.max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_prompt_tuning(self, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_path)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        self.model = self.load_peft_model()
        self.model.print_trainable_parameters()

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        training_args = TrainingArguments("trainer", evaluation_strategy="epoch", num_train_epochs=5, learning_rate=0.003)
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(f"./prompt_tuning/{self.config.kb_name}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")

class BARTEncoderDecoderLM(EncoderDecoderLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = BartForConditionalGeneration.from_pretrained(self.config.model_path, forced_bos_token_id=0)    
        print(f"Loaded BartForConditionalGeneration from {self.config.model_path}")
        self.model.to(self.device)
        self.model.eval()


class BLOOMDecoderLM(EncoderDecoderLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load(self):
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.config.model_path)
        self.model = BloomForCausalLM.from_pretrained(self.config.model_path, device_map="balanced")
        
        if self.config.soft_prompt:
            self.peft_text = self.prompt_tuning_getter.get(self.config.soft_prompt)
            self.model = PeftModel.from_pretrained(self.model, f"{self.config.peft_model_path}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")
            print(f"Loaded BloomForCausalLM from {self.config.model_path} with {self.peft_text}")
        else:
            print(f"Loaded BloomForCausalLM from {self.config.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def output_cleaner(self, pred, **kwargs):
        pred = pred.replace(kwargs['prompt'], "")
        return pred.replace("<pad>", "").replace("</s>", "").strip()

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"{x} " for x in examples[self.text_column]]
        targets = [str(x) for x in examples[self.label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_prompt_tuning(self, dataset):
        self.tokenizer = BloomTokenizerFast.from_pretrained(self.config.model_path)
        self.model = BloomForCausalLM.from_pretrained(self.config.model_path)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        self.model = self.load_peft_model()
        self.model.print_trainable_parameters()

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            "prompt-trainer",
            evaluation_strategy="epoch",
            num_train_epochs=4,
            learning_rate=3e-2
        )
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(f"./prompt_tuning/{self.config.kb_name}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")

class LLaMADecoderLM(EncoderDecoderLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(self.config.model_path, torch_dtype=torch.float16, device_map="balanced")
    
        if self.config.soft_prompt:
            self.peft_text = self.prompt_tuning_getter.get(self.config.soft_prompt)
            self.model = PeftModel.from_pretrained(self.model, f"{self.config.peft_model_path}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")
            print(f"Loaded LLamaForCausalLM from {self.config.model_path} with {self.peft_text}")
        else:
            print(f"Loaded LLamaForCausalLM from {self.config.model_path}")
        
        self.model.to(self.device)
        self.model.eval()

    def output_cleaner(self, pred, **kwargs):
        pred = pred.replace(kwargs['prompt'], "")
        return pred

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"{x} " for x in examples[self.text_column]]
        targets = [str(x) for x in examples[self.label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_prompt_tuning(self, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(self.config.model_path, load_in_8bit=False, torch_dtype=torch.float16, device_map="balanced")
        # self.model = LlamaForCausalLM.from_pretrained(self.config.model_path)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        self.model = self.load_peft_model()
        self.model.print_trainable_parameters()
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            "prompt-trainer",
            evaluation_strategy="epoch",
            num_train_epochs=4,
            learning_rate=3e-2
        )
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(f"./prompt_tuning/{self.config.kb_name}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")

class LLaMA3DecoderLM(EncoderDecoderLM):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, torch_dtype=torch.bfloat16, device_map="balanced")
        
        if self.config.soft_prompt:
            self.peft_text = self.prompt_tuning_getter.get(self.config.soft_prompt)
            self.model = PeftModel.from_pretrained(self.model, f"{self.config.peft_model_path}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")
            print(f"Loaded LLama2-3ForCausalLM from {self.config.model_path} with {self.peft_text}")
        else:
            print(f"Loaded LLama2-3ForCausalLM from {self.config.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def output_cleaner(self, pred, **kwargs):
        pred = pred.replace(kwargs['prompt'], "")
        return pred

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = [f"{x} " for x in examples[self.text_column]]
        targets = [str(x) for x in examples[self.label_column]]
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (self.max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (self.max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:self.max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:self.max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:self.max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_prompt_tuning(self, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, load_in_8bit=False, torch_dtype=torch.float16, device_map="balanced")
        # self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        
        self.model = self.load_peft_model()
        self.model.print_trainable_parameters()
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            "prompt-trainer",
            evaluation_strategy="epoch",
            num_train_epochs=4,
            learning_rate=3e-2
        )
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=processed_datasets["train"],
            eval_dataset=processed_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(f"./prompt_tuning/{self.config.kb_name}/{self.config.soft_prompt}/{self.config.template}/{self.config.model_name}")

# class Left2RightOnlineLM(BaseLM):

#     def __init__(self, config) -> None:
#         super().__init__(config)

#     def output_cleaner(self, pred):
#         return pred.rstrip('\n').strip()

#     def predict(self, X: str):
#         response = openai.Completion.create(
#             model=self.config.model_path,
#             prompt=X,
#             temperature=0.7,
#             max_tokens=self.config.gpt3_max_tokens,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         return response

#     def check_all_is_done(self, results):
#         for result in results:
#             if result['check'] == False:
#                 return False
#         return True

#     def make_batch_prediction(self, Xs):
#         results = []
#         for index, data in enumerate(Xs['sample']):
#             results.append({"check": False})

#         assert self.check_all_is_done(results) == False

#         while not self.check_all_is_done(results):
#             for index, data in tqdm(enumerate(Xs['sample'])):
#                 if results[index]['check'] != True:
#                     try:
#                         response = self.predict(data)
#                         results[index]['result'] = {"response": response, "sample": data, "label": Xs['label'][index]}
#                         results[index]['check'] = True
#                     except Exception as err:
#                         print(f"UNexpected {err}, {type(err)}")
#                         print("Going to sleep for 5 second!")
#                         time.sleep(5)
#         return results


# class GPT4Left2RightOnlineLM(Left2RightOnlineLM):

#     def predict(self, X: str):
#         messages = [{"role": "user", "content": X}]
#         response = openai.ChatCompletion.create(
#             model=self.config.model_path,
#             messages=messages,
#             temperature=0,
#             max_tokens=self.config.gpt4_max_tokens,
#         )
#         return response

# class ChatGPTLeft2RightOnlineLM(Left2RightOnlineLM):

#     def predict(self, X: str):
#         messages = [{"role": "user", "content": X}]
#         response = openai.ChatCompletion.create(
#             model=self.config.model_path,
#             messages=messages,
#             temperature=0,
#             max_tokens=self.config.chatgpt_max_tokens,
#         )
#         return response

class InferenceFactory:

    def __init__(self, config) -> None:
        self.models = {
            "bert_large": MaskedLM,
            "pubmed_bert": MaskedLM,
            "bart_large": BARTMaskedLM,
            "flan_t5_large": FlanT5EncoderDecoderLM,
            "flan_t5_xl": FlanT5EncoderDecoderLM,
            "bloom_1b7": BLOOMDecoderLM,
            "bloom_3b": BLOOMDecoderLM,
            # "gpt3": Left2RightOnlineLM,
            "llama_7b": LLaMADecoderLM,
            # "gpt4": GPT4Left2RightOnlineLM,
            # "chatgpt": ChatGPTLeft2RightOnlineLM,
            "llama3": LLaMA3DecoderLM,
            "llama2": LLaMA3DecoderLM,
            "llama2_chat": LLaMA3DecoderLM,
            "llama3_chat": LLaMA3DecoderLM,
        }
        self.config = config

    def __call__(self, model_name):
        try:
            model = self.models.get(model_name)(config=self.config)
        except ValueError:
            print("Oops! That was not valid model name. Try again ... ")
            exit(0)
        model.load()
        return model

class PromptTuning:
    def __init__(self, config) -> None:
        self.models = {
            "bert_large": MaskedLM,
            "pubmed_bert": MaskedLM,
            "bart_large": BARTMaskedLM,
            "flan_t5_large": FlanT5EncoderDecoderLM,
            "flan_t5_xl": FlanT5EncoderDecoderLM,
            "bloom_1b7": BLOOMDecoderLM,
            "bloom_3b": BLOOMDecoderLM,
            "llama_7b": LLaMADecoderLM,
            "llama3": LLaMA3DecoderLM,
            "llama2": LLaMA3DecoderLM,
            "llama2_chat": LLaMA3DecoderLM,
            "llama3_chat": LLaMA3DecoderLM,
        }
        self.config = config

    def __call__(self, model_name, dataset):
        try:
            model = self.models.get(model_name)(config=self.config)
        except ValueError:
            print("Oops! That was not valid model name. Try again ... ")
            exit(0)
        model.prepare_prompt_tuning(dataset)
        return model

