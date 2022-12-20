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

class CalibrationMetric(Callback):
    def __init__(self, monitor="all", correct=[], y_prob_max=[], num_bins=10):
        self.total = 0
        self.monitor = monitor
        self.sanity = False

        self.correct = correct
        self.y_prob_max = y_prob_max
        self.num_bins = num_bins # how do we make this an user-specified input?

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
  
        self.y_prob_max += np.amax(outputs["p2u_outputs"]["raw"].logits.cpu().numpy(), axis=-1).tolist()
        self.correct += (np.argmax(outputs["p2u_outputs"]["raw"].logits.cpu().numpy(), axis=1) == outputs["targets"]["label"].cpu().numpy()).astype(int).tolist()

    def on_test_epoch_end(self, trainer, pl_module):

        bins = np.linspace(0., 1. + 1e-8, self.num_bins + 1)
        bin_ids = np.digitize(self.y_prob_max, bins) - 1


        bin_sums = np.bincount(bin_ids, weights=self.y_prob_max, minlength=len(bins))
        bin_true = np.bincount(bin_ids, weights=self.correct, minlength=len(bins))
        bin_total = np.bincount(bin_ids, minlength=len(bins))

        non_zero = bin_total != 0
        prob_true = bin_true[non_zero] / bin_total[non_zero]
        prob_pred = bin_sums[non_zero] / bin_total[non_zero]

        expected_calibration_error = np.sum(bin_total[non_zero] * np.abs(prob_true - prob_pred))/bin_total[non_zero].sum()
        print('Expected Calibration Error ----', expected_calibration_error)
        logging.info(
            f"The expected calibration error for {self.monitor} is: {expected_calibration_error*100}%"
        )