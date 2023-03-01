import logging
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import AutoTokenizer, T5Tokenizer


class mnli_tokenization:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        is_generative_model: bool,
        tokenizer_args: dict,
        data_processing: dict,
        label2id: dict,
        cols: list,
        label_col: str,
    ):
        if model_type != "t5":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.is_generative_model = is_generative_model
        self.data_processing = data_processing
        self.tokenizer_args = tokenizer_args
        self.label2id = label2id
        self.label_col = label_col
        self.cols = cols

    def process(self, example):
        return self.tokenizer(example["input_data"], **self.tokenizer_args)


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
        model_type: str,
        data_dir: str = "data/",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        augmentations: list = [],
        data_processing: dict = {},
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
        self.data_processing = data_processing
        self.is_generative_model = False if model_type == "discriminative" else True

        self.tokenization = mnli_tokenization(
            model_name=self.tokenizer_data["model_name"],
            is_generative_model=self.is_generative_model,
            tokenizer_args=self.tokenizer_data["args"],
            data_processing=self.data_processing,
            label2id=self.label2id,
            model_type=model_type,
            label_col="label",
            cols=self.cols,
        )

        self.augmentations = augmentations

    @property
    def num_classes(self) -> int:
        return len(self.label2id)

    def perform_augmentations(self):
        logging.info(f"Before augmentation dataset length: {len(self.data_test)}")

        self.augmentations = [aug(dataset=self.data_test) for aug in self.augmentations]
        self.augmentations = [aug.get_dataset() for aug in self.augmentations]
        self.data_test = concatenate_datasets(self.augmentations)
        # self.data_test = concatenate_datasets([self.data_test, self.augmented_data])

        logging.info(f"After augmentation dataset length: {len(self.data_test)}")

    def custom_prepocess(self, dataset):
        if self.data_processing.columns:
            for column_name, column_prefix in self.data_processing.columns.items():
                dataset = dataset.map(
                    lambda example: {column_name: " ".join([column_prefix, example[column_name]])},
                    batched=False,
                )

        dataset = dataset.map(
            lambda example: {
                "input_data": self.data_processing.separator.join(
                    [example[col] for col in self.cols]
                )
            },
            remove_columns=self.cols,
            batched=False,
        )

        if self.data_processing.header:
            dataset = dataset.map(
                lambda example: {
                    "input_data": self.data_processing.separator.join(
                        [self.data_processing.header] + [example["input_data"]]
                    )
                },
                batched=False,
            )

        if self.data_processing.footer:
            dataset = dataset.map(
                lambda example: {
                    "input_data": self.data_processing.separator.join(
                        [example["input_data"]] + [self.data_processing.footer]
                    )
                },
                batched=False,
            )

        return dataset

    def prepare_data(self):
        self.data_test = load_dataset("multi_nli", split="validation_matched")
        self.data_test = self.data_test.remove_columns(["promptID", "pairID"])

        logging.info("Performing data augmentations...")
        self.perform_augmentations()

        logging.info("Pre-processing the data...")
        self.data_test = self.custom_prepocess(dataset=self.data_test)

        logging.info("Performing tokenization...")
        old_columns = set(list(self.data_test.features.keys()))
        self.data_test = self.data_test.map(self.tokenization.process, batched=True)
        self.label_conversion = process_label2id(self.label2id, self.tokenizer_data.label2id)
        self.data_test = self.data_test.map(
            lambda batch: {"converted_label": self.label_conversion[batch["label"]]},
            batched=False,
            remove_columns=["label"],
        )
        self.data_test = self.data_test.rename_column("converted_label", "label")

        if self.is_generative_model:
            self.data_test = self.data_test.map(
                lambda batch: {
                    "converted_label": self.tokenization.tokenizer.convert_tokens_to_ids(
                        batch["label"]
                    )
                },
                batched=False,
                remove_columns=["label"],
            )
            self.data_test = self.data_test.rename_column("converted_label", "label")

        new_columns = set(list(self.data_test.features.keys()))

        self.data_test.set_format(
            type="torch",
            columns=["label", "augmentation"] + list(new_columns - old_columns),
        )
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
