from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm 
import pandas as pd
from datasets import ClassLabel, Dataset

class back_translate_augmentation:
    def __init__(self, cols=None):
        
        self.cols = cols
        self.model_translate_name = "Helsinki-NLP/opus-mt-en-roa"
        self.model_back_translate_name = "Helsinki-NLP/opus-mt-roa-en"

    def download(self, model_name):
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer

    def infer(self, dataset):
        
        model_translate, model_translate_tokenizer = self.download(self.model_translate_name)
        model_back_translate, model_back_translate_tokenizer = self.download(self.model_back_translate_name)

        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}

        for i in tqdm(range(len(dataset))):
            src_text = [">>fra<< " + dataset[i]["text"]]
            src_translated = model_translate.generate(**model_translate_tokenizer(src_text, return_tensors="pt", padding=True))
            
            tgt_text = [model_translate_tokenizer.decode(t, skip_special_tokens=True) for t in src_translated][0]
            tgt_text = [">>eng<< " + tgt_text]
            tgt_translated = model_back_translate.generate(**model_back_translate_tokenizer(tgt_text, return_tensors="pt", padding=True))
            
            back_translated_text = [model_back_translate_tokenizer.decode(t, skip_special_tokens=True) for t in tgt_translated][0] 
            if(back_translated_text != dataset[i]["text"]):
                new_dataset["text"] = back_translated_text
                new_dataset["label"].append(dataset["label"][i])
                new_dataset["mapping"].append(i)
                for k in datacols:
                    if k not in ["label", "mapping"] + self.cols:
                        new_dataset[k].append(dataset[k][i])
        
        new_dataset = pd.DataFrame(new_dataset)
        return Dataset.from_pandas(new_dataset).cast_column("label", dataset.features["label"])

    
    