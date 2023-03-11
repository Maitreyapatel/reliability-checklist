import random 
from tqdm import tqdm
import pandas as pd
from datasets import ClassLabel, Dataset
import numpy as np 

LOWER_YEAR_NUM = 1000
UPPER_YEAR_NUM = 2020

class num_word_augmentation:
    def __init__(self):
        pass
    def infer(self, dataset, n_workers="max"):
        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}
        for i in tqdm(range(len(dataset))):
            premise_tokens = dataset["premise"][i].split()
            flag = False
            count_num = 0 
            for num, token in enumerate(premise_tokens):
                if (token.isdigit()):
                    number = int(token)
                    count_num += 1
                    if LOWER_YEAR_NUM <= number <= UPPER_YEAR_NUM:
                        continue
                    cont_hyp = get_contradictory_hypothesis(premise_tokens, num, number)
                    flag = True
                    break
            if flag and count_num == 1:
                new_dataset["hypothesis"].append(dataset["premise"][i])
                new_dataset["premise"].append(cont_hyp)
                new_dataset["label"].append(2)
                new_dataset["mapping"].append(i)
                for k in datacols:
                    if k not in ["premise", "hypothesis", "label", "mapping"]:
                        new_dataset[k].append(dataset[k][i])
        
        new_dataset = pd.DataFrame(new_dataset)
        return Dataset.from_pandas(new_dataset).cast_column(
            "label",
            ClassLabel(num_classes=3, names=["entailment", "neutral", "contradiction"]),
        )

def get_contradictory_hypothesis(tokens, index, number):

    prob = np.random.binomial(1, 0.5)

    if prob < 0.5:
        number = str(number)
        new_digit = np.random.randint(1, 9)
        old_digit = int(number[0])
        while new_digit == old_digit:
            new_digit = np.random.randint(1, 9)
        new_num = str(new_digit) + number[1:]
        new_tokens = tokens[:index] + [new_num] + tokens[index + 1:]
    else:
        prob2 = np.random.binomial(1, 0.5)
        if prob2 < 0.5:
            new_tokens = tokens[:index] + \
                ['more than', str(number)] + tokens[index + 1:]
        else:
            new_tokens = tokens[:index] + \
                ['less than', str(number)] + tokens[index + 1:]

    return ' '.join(new_tokens)
