import random
from random import SystemRandom

import pandas as pd
from datasets import ClassLabel, Dataset
from tqdm import tqdm


class rand_sentence_augmentation:
    def __init__(self):
        self.IRRELEVANT_SENTENCES = [
            "I prefer tea over coffee.",
            "The sky is blue on a clear day.",
            "I need to buy some new socks.",
            "My favorite color is purple.",
            "The Earth orbits around the sun.",
            "I enjoy watching movies in my free time.",
            "I have a dog named Max.",
            "The capital of France is Paris.",
            "I am allergic to peanuts.",
            "I like to listen to music when I work.",
            "I have never been skydiving.",
            "The Great Barrier Reef is the world's largest coral reef system.",
            "I enjoy hiking in the mountains.",
            "I am currently reading a book about history.",
            "My favorite food is sushi.",
            "I can speak two languages fluently.",
            "The Mona Lisa is a famous painting by Leonardo da Vinci.",
            "I am not a morning person.",
            "I have visited several countries in Europe.",
            "I like to go for a run in the morning.",
        ]

    def infer(self, dataset, n_workers="max"):
        cryptogen = SystemRandom()

        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}
        for i in tqdm(range(len(dataset))):
            if cryptogen.uniform(0, 1) < 0.5:
                new_hypothesis = (
                    cryptogen.choice(self.IRRELEVANT_SENTENCES) + " " + dataset["hypothesis"][i]
                )
            else:
                new_hypothesis = (
                    dataset["hypothesis"][i] + " " + cryptogen.choice(self.IRRELEVANT_SENTENCES)
                )
            new_dataset["hypothesis"].append(new_hypothesis)
            new_dataset["premise"].append(dataset["premise"][i])
            new_dataset["label"].append(dataset["label"][i])
            new_dataset["mapping"].append(i)
            for k in datacols:
                if k not in ["premise", "hypothesis", "label", "mapping"]:
                    new_dataset[k].append(dataset[k][i])
        new_dataset = pd.DataFrame(new_dataset)
        return Dataset.from_pandas(new_dataset).cast_column("label", dataset.features["label"])
