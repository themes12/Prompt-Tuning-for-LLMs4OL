from torch.utils.data import Dataset

class WNDataset(Dataset):
    def __init__(self, data, templates, template, is_train, prompt_tune):
        self.data = data
        self.template = templates[template]
        self.is_train = is_train
        self.dataset_type = "train" if self.is_train else "test"
        self.use_sentence =  True if "[SENTENCE]" in self.template else False
        self.prompt_tune = prompt_tune
        print(f"WNDataset:{'Train-SET' if self.is_train else 'Test-SET'} --- {template}: {self.template} size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
         return f"WNDataset:{'Train-SET' if self.is_train else 'Test-SET'} --- Template-in-use: {self.template} size: {len(self.data)}"
    
    def __getitem__(self, index):
        item = self.data[index]
        sample, label = self.template.replace("[A]", item['entity']), item['label']
        label = label if self.prompt_tune else [label]
        if self.use_sentence:
            sample = sample.replace("[SENTENCE]", item['sentence'])
        if self.is_train:
            sample = sample.replace("[MASK]", item['label'])
        return {"ID": item["original-entity"], "sample":sample, "label":label}

    def collate_fn(self, batchs):
        batchs_clear = {"ID": [], "sample":[], "label":[]}
        for batch in batchs:
            batchs_clear['ID'].append(batch['ID'])
            batchs_clear['sample'].append(batch['sample'])
            batchs_clear['label'].append(batch['label'])
        return batchs_clear

class GeonameDataset(Dataset):
    def __init__(self, data, kb_name, templates, template, is_train, label_mapper, prompt_tune):
        self.template = templates[template]
        self.is_train = is_train
        self.label_mapper = label_mapper
        self.use_country = True if "[COUNTRY]" in self.template else False
        self.data = []
        self.prompt_tune = prompt_tune
        for sample in data:
        # for sample in data:
            if self.is_train and sample['status'] == "train":
                self.data.append(sample)
            if not self.is_train and sample['status'] == "test":
                self.data.append(sample)
            # if self.is_train:
            #     self.data.append(sample)
            # if not self.is_train:
            #     self.data.append(sample)
        # self.data = self.data[:100]
        print(len(self.data))
        print(f"GeonamesDataset:{'Train-SET' if self.is_train else 'Test-SET'} --- {template}: {self.template} size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
         return f"GeonamesDataset:{'Train-SET' if self.is_train else 'Test-SET'} --- Template-in-use: {self.template} size: {len(self.data)}"
    
    def __getitem__(self, index):
        item = self.data[index]
        sample, label = self.template.replace("[A]", str(item['asciname'])), item['type-label']
        labels = [self.label_mapper[label]['name']]+self.label_mapper[label]['synonyms']
        label = list(set(labels))
        label = [l.lower() for l in label]
        if self.use_country:
            sample = sample.replace("[COUNTRY]", str(item["country_name"]))
        if self.is_train:
            sample = sample.replace("[MASK]", 'or '.join(item['type-name']))
        return {"ID": item["ID"], "sample":sample, "label":label}

    def collate_fn(self, batchs):
        batchs_clear = {"ID": [], "sample":[], "label":[]}
        for batch in batchs:
            batchs_clear['ID'].append(batch['ID'])
            batchs_clear['sample'].append(batch['sample'])
            batchs_clear['label'].append(batch['label'])
        return batchs_clear

class UMLSDataset(Dataset):
    def __init__(self, data, kb_name, templates, template, is_train, label_mapper, prompt_tune):
        self.template = templates[template]
        self.is_train = is_train
        self.kb_name = kb_name
        self.label_mapper = label_mapper
        self.use_sentence =  True if "[SENTENCE]" in self.template else False
        self.data = []
        self.prompt_tune = prompt_tune
        for sample in data:
            if self.is_train and sample["status"] == "train":
                self.data.append(sample)
            if not self.is_train and sample["status"] == "test":
                self.data.append(sample)
        print(f"UMLSDataset:{'Train-SET' if self.is_train else 'Test-SET'}-{self.kb_name} --- {template}: {self.template} size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
         return f"UMLSDataset:{'Train-SET' if self.is_train else 'Test-SET'}-{self.kb_name} --- Template-in-use: {self.template} size: {len(self.data)}"
    
    def __getitem__(self, index):
        item = self.data[index]
        concept = str(item['concept']).lower()
        sample = self.template.replace("[A]", concept)
        labels = []
        for label in item['label-str']:
        # for label in eval(item['label-str']):
            for l in self.label_mapper[label]:
                labels.append(l)
        label = list(set(labels))
        label = [l.lower() for l in label]
        if self.use_sentence:
            sample = sample.replace("[SENTENCE]", concept)
        if self.is_train:
            sample = sample # .replace("[MASK]", 'or '.join(item['label-names']))
        return {"ID": item["ID"], "sample":sample, "label":label}

    def collate_fn(self, batchs):
        batchs_clear = {"ID": [], "sample":[], "label":[]}
        for batch in batchs:
            batchs_clear['ID'].append(batch['ID'])
            batchs_clear['sample'].append(batch['sample'])
            batchs_clear['label'].append(batch['label'])
        return batchs_clear
    
class GODataset(Dataset):
    def __init__(self, data, kb_name, templates, template, is_train, prompt_tune):
        self.template = templates[template]
        self.is_train = is_train
        self.kb_name = kb_name
        # self.label_mapper = label_mapper
        self.use_sentence =  True if "[SENTENCE]" in self.template else False
        self.data = []
        self.prompt_tune = prompt_tune
        for sample in data:
            if self.is_train and sample["status"] == "train":
                self.data.append(sample)
            if not self.is_train and sample["status"] == "test":
                self.data.append(sample)
        print(f"GODataset:{'Train-SET' if self.is_train else 'Test-SET'}-{self.kb_name} --- {template}: {self.template} size: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self) -> str:
         return f"GODataset:{'Train-SET' if self.is_train else 'Test-SET'}-{self.kb_name} --- Template-in-use: {self.template} size: {len(self.data)}"
    
    def __getitem__(self, index):
        item = self.data[index]
        concept = str(item['term']).lower()
        sample = self.template.replace("[A]", concept)
        
        if self.kb_name == "biological":
            sample = sample.replace("[KB]", "biological process")
        if self.kb_name == "molecular":
            sample = sample.replace("[KB]", "molecular function")

        labels = item["type"]
        # for label in item['label-str']:
        # for label in eval(item['label-str']):
            # for l in self.label_mapper[label]:
                # labels.append(l)
        # label = list(set(labels))
        label = [l.lower().replace("_", " ") for l in labels]
        if self.use_sentence:
            sample = sample.replace("[SENTENCE]", concept)
        if self.is_train:
            sample = sample # .replace("[MASK]", 'or '.join(item['label-names']))
        return {"ID": item["ID"], "sample":sample, "label":label}

    def collate_fn(self, batchs):
        batchs_clear = {"ID": [],"sample":[], "label":[]}
        for batch in batchs:
            batchs_clear['ID'].append(batch['ID'])
            batchs_clear['sample'].append(batch['sample'])
            batchs_clear['label'].append(batch['label'])
        return batchs_clear


class InferenceDatasetFactory:
    def __new__(CLS, kb_name, data, templates, template, label_mapper, prompt_tune=False, is_train=False) -> Dataset:
        if kb_name == "geonames":
            return GeonameDataset(data=data, kb_name=kb_name, 
                                  templates=templates, template=template,
                                  is_train=is_train, label_mapper=label_mapper, prompt_tune=prompt_tune)

        if kb_name == "wn18rr":
            return WNDataset(data=data, templates=templates,
                             template=template, is_train=is_train, prompt_tune=prompt_tune)

        if kb_name == "nci" or kb_name == "snomedct_us" or kb_name == "medcin":
            return UMLSDataset(data=data, kb_name=kb_name, 
                                templates=templates, template=template,
                                is_train=is_train, label_mapper=label_mapper, prompt_tune=prompt_tune)
        
        if kb_name == "biological" or kb_name == "cellular" or kb_name == "molecular":
            return GODataset(data=data, kb_name=kb_name,
                                templates=templates, template=template,
                                is_train=is_train, prompt_tune=prompt_tune)
