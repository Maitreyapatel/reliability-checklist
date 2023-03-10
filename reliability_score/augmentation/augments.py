import logging
import os
from copy import deepcopy

import pandas as pd
from datasets import ClassLabel, Dataset
from tqdm import tqdm

from reliability_score.augmentation.mnli.augmentation import nli_augmentations


class Augmentation:
    def __init__(self, __name__, dataset=None):
        self.__name__ = __name__
        self.dataset = dataset

    def add_name_col(self):
        new_column = [self.__name__] * len(self.dataset)
        self.dataset = self.dataset.add_column("augmentation", new_column)
        return self.dataset

    def augment(self):
        # write your custom augmentation script
        raise NotImplementedError

    def get_dataset(self):
        self.augment()
        self.add_name_col()
        return self.dataset


class NoAug(Augmentation):
    def __init__(self, __name__="DEFAULT", dataset=None):
        super().__init__(__name__=__name__, dataset=dataset)

    def augment(self):
        pass


class DummyAug(Augmentation):
    def __init__(self, __name__="DUMMY", dataset=None):
        super().__init__(__name__=__name__, dataset=dataset)

    def augment(self):
        pass


class simple_aug(Augmentation):
    def __init__(self, __name__="INV_PASS", dataset=None):
        super().__init__(__name__, dataset)
        self.augmenter = nli_augmentations()

    def augment(self):
        self.dataset = self.augmenter.infer(self.dataset)


class parrot_paraphraser(Augmentation):
    def __init__(
        self,
        __name__="parrot",
        dataset=None,
        dataset_name="none",
        csv_path=None,
        cols=None,
    ):
        super().__init__(__name__, dataset)
        self.csv_path = csv_path.format(dataset_name)
        self.cols = cols

        from parrot import Parrot

        self.parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    def perform_augmentation(self, dataset):
        logging.warn("Parrot data augmentation initiated. This is a very slow process!")
        datacols = list(dataset.features.keys()) + ["mapping"]
        new_dataset = {k: [] for k in datacols}

        for i in tqdm(range(len(dataset))):
            tmp_data = {}
            tmp_flag = True
            tmp_min = 10  # fixed maximum parrot augmentations
            for col in self.cols:
                tmp_ = self.parrot.augment(input_phrase=dataset[col][i], use_gpu=True)
                tmp_data[col] = tmp_
                if not tmp_:
                    tmp_flag = False
                else:
                    tmp_min = min(tmp_min, len(tmp_))

            if tmp_flag:
                for j in range(tmp_min):
                    for col in self.cols:
                        new_dataset[col].append(tmp_data[col][0])
                    new_dataset["label"].append(dataset["label"][i])
                    new_dataset["mapping"].append(i)
                    for k in datacols:
                        if k not in ["label", "mapping"] + self.cols:
                            new_dataset[k].append(dataset[k][i])

        new_dataset = pd.DataFrame(new_dataset)
        new_dataset.to_csv(self.csv_path, sep="\t")

    def augment(self):
        if not os.path.exists(self.csv_path):
            logging.warn(f"Could not find the pre-defined csv data file at: {self.csv_path}")
            self.perform_augmentation(self.dataset)

        new_dataset = Dataset.from_pandas(pd.read_csv(self.csv_path, delimiter="\t"))
        self.dataset = new_dataset.cast_column(
            "label",
            self.dataset.features["label"],
        )
