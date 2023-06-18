import pandas as pd
from datasets import ClassLabel, Dataset
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


class double_denial_augmentation_sst2:
    def __init__(self, cols=None):
        self.cols = cols
        self.polarity_dict = {
            "poor": "not good",
            "bad": "not great",
            "lame": "not interesting",
            "awful": "not awesome",
            "great": "not bad",
            "good": "not poor",
            "applause": "not discourage",
            "recommend": "don't prevent",
            "best": "not worst",
            "encourage": "don't discourage",
            "entertain": "don't disapprove",
            "wonderfully": "not poorly",
            "love": "don't hate",
            "interesting": "not uninteresting",
            "interested": "not ignorant",
            "glad": "not reluctant",
            "positive": "not negative",
            "perfect": "not imperfect",
            "entertaining": "not uninteresting",
            "moved": "not moved",
            "like": "don't refuse",
            "worth": "not undeserving",
            "better": "not worse",
            "funny": "not uninteresting",
            "awesome": "not ugly",
            "impressed": "not impressed",
        }

    def infer(self, dataset):
        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}

        for i in tqdm(range(len(dataset))):
            flag = False
            tokens = dataset[i]["sentence"].split()
            augmented_string = ""
            for each_token in tokens:
                if each_token in self.polarity_dict:
                    augmented_string += self.polarity_dict[each_token]
                    flag = True
                else:
                    augmented_string += each_token
                augmented_string += " "

            if flag:
                new_dataset["sentence"].append(augmented_string)
                new_dataset["label"].append(dataset["label"][i])
                new_dataset["mapping"].append(i)

                for k in datacols:
                    if k not in ["label", "mapping"] + self.cols:
                        new_dataset[k].append(dataset[k][i])

        new_dataset = pd.DataFrame(new_dataset)
        return Dataset.from_pandas(new_dataset).cast_column("label", dataset.features["label"])
