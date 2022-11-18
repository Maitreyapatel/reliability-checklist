from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from datasets import load_dataset
from transformers import AutoTokenizer

class mnli_tokenization:
    def __init__(self, model_name: str, tokenizer_args: dict, cols: list):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_args = tokenizer_args
        self.cols = cols

    def process(self, example):
        return self.tokenizer(example[self.cols[0]], example[self.cols[1]], **self.tokenizer_args)


class MNLIDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_data: dict,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.cols = ["premise", "hypothesis"]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir

        self.data_test: Optional[Dataset] = None
        self.tokenizer_data = tokenizer_data
        self.tokenization = mnli_tokenization(model_name = self.tokenizer_data['model_name'], tokenizer_args=self.tokenizer_data['args'], cols=self.cols)

    @property
    def num_classes(self) -> int:
        return len(self.label2id)

    def prepare_data(self):
        self.data_test = load_dataset("glue", "mnli", split='validation_matched')
        self.data_test = self.data_test.map(self.tokenization.process, batched=True)

        self.data_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
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
