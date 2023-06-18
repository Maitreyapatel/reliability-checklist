import random

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset
from nltk.wsd import lesk
from tqdm import tqdm


class swap_antonym_aug_qnli:
    def __init__(self):
        pass

    def infer(self, dataset, n_workers="max"):
        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}

        for i in tqdm(range(len(dataset))):
            if dataset["label"][i] == 0:
                premise_tokens = dataset["question"][i].split()
                hypothesis_tokens = dataset["sentence"][i].split()
                new_hyp = dataset["question"][i]
                flag = False
                for num, pr_token in enumerate(premise_tokens):
                    best_sense = lesk(premise_tokens, pr_token)
                    if best_sense is not None and (
                        best_sense.pos() == "s" or best_sense.pos() == "n"
                    ):
                        for lemma in best_sense.lemmas():
                            possible_antonyms = lemma.antonyms()
                            for antonym in possible_antonyms:
                                if "_" in antonym._name or antonym._name == "civilian":
                                    continue
                                if pr_token not in hypothesis_tokens:
                                    continue
                                new_hyp = new_hyp.replace(pr_token, antonym._name)
                                flag = True

                if flag:
                    new_dataset["sentence"].append(new_hyp)
                    new_dataset["question"].append(dataset["question"][i])
                    new_dataset["label"].append(1)
                    new_dataset["mapping"].append(i)
                    for k in datacols:
                        if k not in ["sentence", "question", "label", "mapping"]:
                            new_dataset[k].append(dataset[k][i])

        new_dataset = pd.DataFrame(new_dataset)
        return Dataset.from_pandas(new_dataset).cast_column("label", dataset.features["label"])
