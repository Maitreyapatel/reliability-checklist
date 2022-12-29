from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset, concatenate_datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import AutoTokenizer

import logging

class mnli_tokenization:
    def __init__(self, model_name: str, tokenizer_args: dict, cols: list):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_args = tokenizer_args
        self.cols = cols

    def process(self, example):
        return self.tokenizer(example[self.cols[0]], example[self.cols[1]], **self.tokenizer_args)


def process_label2id(gt_label2id, pred_label2id):
    assert len(gt_label2id) == len(pred_label2id)

    dataset_converion = {}
    for i in list(gt_label2id.keys()):
        dataset_converion[i] = pred_label2id[gt_label2id[i]]
    return dataset_converion


class MNLIDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_data: dict,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        augmentations: list = [],
    ):
        super().__init__()

        self.label2id = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.cols = ["premise", "hypothesis"]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir

        self.data_test: Optional[Dataset] = None
        self.tokenizer_data = tokenizer_data
        self.tokenization = mnli_tokenization(
            model_name=self.tokenizer_data["model_name"],
            tokenizer_args=self.tokenizer_data["args"],
            cols=self.cols,
        )
        self.label_conversion = process_label2id(self.label2id, tokenizer_data.label2id)
        
        self.augmentations = augmentations

    @property
    def num_classes(self) -> int:
        return len(self.label2id)
    
    def perform_augmentations(self):
        logging.info(f"Before augmentation dataset length: {len(self.data_test)}")

        self.augmentations = [aug(dataset=self.data_test) for aug in self.augmentations]
        self.augmentations = [aug.get_dataset() for aug in self.augmentations]
        self.augmented_data = concatenate_datasets(self.augmentations)
        self.data_test = concatenate_datasets([self.data_test, self.augmented_data])
        
        logging.info(f"After augmentation dataset length: {len(self.data_test)}")

    def prepare_data(self):
        self.data_test = load_dataset("glue", "mnli", split="validation_matched")
        old_columns = set(list(self.data_test.features.keys()))
        self.data_test = self.data_test.map(self.tokenization.process, batched=True)
        self.data_test = self.data_test.map(
            lambda batch: {"label": self.label_conversion[batch["label"]]}, batched=False
        )
        new_columns = set(list(self.data_test.features.keys()))

        self.data_test.set_format(
            type="torch", columns=["label"] + list(new_columns - old_columns)
        )
        
        logging.info("Performing data augmentations...")
        self.perform_augmentations()
        # self.data_test = self.data_test.align_labels_with_mapping(self.label2id, "label")

    def setup(self, stage: Optional[str] = None):
        if not self.data_test:
            raise

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnli.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
