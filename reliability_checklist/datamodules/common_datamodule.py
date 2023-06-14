import logging
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from reliability_checklist.datamodules.utils import (
    conversion_process,
    general_tokenization,
    process_label2id,
)


class GeneralDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_specific_args: dict,
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

        self.label2id = dataset_specific_args["label2id"]
        self.cols = dataset_specific_args["cols"]
        self.dataset_parent = dataset_specific_args["parent"]
        self.dataset_name = dataset_specific_args["name"]
        self.dataset_split = dataset_specific_args["split"]
        self.dataset_rmcols = dataset_specific_args["remove_cols"]
        self.label_col = dataset_specific_args["label_col"]
        self.label_conversion = dataset_specific_args["label_conversion"]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir

        self.data_test: Optional[Dataset] = None
        self.tokenizer_data = tokenizer_data
        self.data_processing = data_processing
        self.is_generative_model = False if model_type == "discriminative" else True

        self.tokenization = general_tokenization(
            model_name=self.tokenizer_data["model_name"],
            is_generative_model=self.is_generative_model,
            tokenizer_args=self.tokenizer_data["args"],
            data_processing=self.data_processing,
            label2id=self.label2id,
            model_type=model_type,
            label_col=self.label_col,
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
        if not self.dataset_parent:
            self.data_test = load_dataset(self.dataset_name, split=self.dataset_split)
        else:
            self.data_test = load_dataset(self.dataset_parent, self.dataset_name, split=self.dataset_split)
        self.data_test = self.data_test.remove_columns(self.dataset_rmcols)
        if self.label_col != "label":
            self.data_test = self.data_test.rename_column(self.label_col, "label")
        if self.label_conversion:
            self.data_test = self.data_test.map(
                lambda batch: {"converted_label": self.label_conversion[batch["label"]]},
                batched=False,
                remove_columns=["label"],
            )
            self.data_test = self.data_test.rename_column("converted_label", "label")
            # cvp = conversion_process(self.label_conversion)
            # self.data_test.map(cvp.process, batched=True)

        keys = [i for i in range(len(self.data_test))]
        self.data_test = self.data_test.add_column("primary_key", keys)

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
            columns=[
                k
                for k in ["label", "augmentation", "mapping", "primary_key"]
                if k in self.data_test.column_names
            ]
            + list(new_columns - old_columns),
        )

    def setup(self, stage: Optional[str] = None):
        if not self.data_test:
            logging.error("It seems that dataset object was not declared. Attempting it again.")
            self.prepare_data()

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
