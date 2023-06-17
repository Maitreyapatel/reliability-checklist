import random
from difflib import SequenceMatcher

import pandas as pd
from datasets import ClassLabel, Dataset
from tqdm import tqdm


class maximum_string_match_aug_qqp:
    def __init__(self):
        pass

    def infer(self, dataset, n_workers="max"):

        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}
        for i in tqdm(range(len(dataset))):
            
            longest_match = SequenceMatcher(None, dataset["question1"][i], dataset["question2"][i]).find_longest_match()
            if(longest_match.size > 5):
                question1 = dataset["question1"][i][:longest_match.a] + dataset["question1"][i][longest_match.a+longest_match.size:]
                if(len(question1) > 0) :
                    if question1[-1] != '?':
                        question1 += '?'
                    new_dataset["question1"].append(question1)
                    
                    new_dataset["question2"].append(dataset["question2"][i])
                    new_dataset["label"].append(0)
                    new_dataset["mapping"].append(i)
                    for k in datacols:
                        if k not in ["question1", "question2", "label", "mapping"]:
                            new_dataset[k].append(dataset[k][i])
        new_dataset = pd.DataFrame(new_dataset)
        return Dataset.from_pandas(new_dataset).cast_column("label", dataset.features["label"])
