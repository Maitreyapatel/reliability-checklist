import logging

import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback


class AccuracyMetric(Callback):
    def __init__(self, results_dir="", monitor="all"):
        self.results_dir = results_dir

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
        with open(os.path.join(self.results_dir, 'acc.txt'), 'w') as h:
            h.write(f"The predicted accuracy for {self.monitor} is: {self.correct*100/self.total}% \n")

        self.total = 0
        self.correct = 0


class CalibrationMetric(Callback):
    def __init__(self, results_dir="", monitor="all", num_bins=10):
        self.results_dir = results_dir
        self.monitor = monitor
        self.sanity = False

        self.correct = []
        self.y_prob_max = []
        self.num_bins = num_bins  # how do we make this an user-specified input?

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        self.y_prob_max += np.amax(
            outputs["p2u_outputs"]["raw"].logits.softmax(dim=1).cpu().numpy(), axis=-1
        ).tolist()
        self.correct += (
            (
                np.argmax(outputs["p2u_outputs"]["raw"].logits.cpu().numpy(), axis=1)
                == outputs["targets"]["label"].cpu().numpy()
            )
            .astype(int)
            .tolist()
        )

    def on_test_epoch_end(self, trainer, pl_module):

        bins = np.linspace(0.0, 1.0 + 1e-8, self.num_bins + 1)
        bin_ids = np.digitize(self.y_prob_max, bins) - 1

        bin_sums = np.bincount(bin_ids, weights=self.y_prob_max, minlength=len(bins))
        bin_true = np.bincount(bin_ids, weights=self.correct, minlength=len(bins))
        bin_total = np.bincount(bin_ids, minlength=len(bins))

        non_zero = bin_total != 0
        prob_true = bin_true[non_zero] / bin_total[non_zero]
        prob_pred = bin_sums[non_zero] / bin_total[non_zero]

        expected_calibration_error = (
            np.sum(bin_total[non_zero] * np.abs(prob_true - prob_pred)) / bin_total[non_zero].sum()
        )
        logging.info(
            f"The expected calibration error for {self.monitor} is: {expected_calibration_error*100}%"
        )

        overconfidence_error = np.sum(
            bin_total[non_zero]
            * prob_pred
            * np.max(
                np.concatenate(
                    ((prob_pred - prob_true).reshape(-1, 1), np.zeros((1, len(prob_pred))).T),
                    axis=1,
                ),
                axis=-1,
            )
            / bin_total[non_zero].sum()
        )
        logging.info(
            f"The overconfidence error for {self.monitor} is: {overconfidence_error*100}%"
        )

        with open(os.path.join(self.results_dir, 'calibration.txt'), 'w') as h:
            h.write(f"The overconfidence error for {self.monitor} is: {overconfidence_error*100}% \n")
            h.write(f"The expected calibration error for {self.monitor} is: {expected_calibration_error*100}% \n")

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        ax1.plot(prob_pred, prob_true, "s-", label="$dataset_name")

        ax1.set_xlabel("Confidence Probability")
        ax1.set_ylabel("Accuracy")
        ax1.legend(loc="lower right")
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_title("Calibration Plot")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "calibration.png"), bbox_inches="tight")
