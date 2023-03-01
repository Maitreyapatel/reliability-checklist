from copy import deepcopy

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
