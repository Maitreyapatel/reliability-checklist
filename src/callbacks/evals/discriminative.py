import logging

import numpy as np
from pytorch_lightning.callbacks import Callback


class AccuracyMetric(Callback):
    def __init__(self, monitor="all"):
        self.total = 0
        self.correct = 0
        self.monitor = monitor
        self.sanity = False

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        self.total += len(outputs["p2u_outputs"]["raw"])
        self.correct += np.sum(
            np.argmax(outputs["p2u_outputs"]["raw"].logits.cpu().numpy(), axis=1)
            == outputs["targets"]["label"].cpu().numpy()
        )

    def on_test_epoch_end(self, trainer, pl_module):

        logging.info(
            f"The predicted accuracy for {self.monitor} is: {self.correct*100/self.total}%"
        )

        self.total = 0
        self.correct = 0
