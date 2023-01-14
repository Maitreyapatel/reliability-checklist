import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback


class MonitorBasedMetric(Callback):
    def __init__(self, monitor, name, results_dir) -> None:
        self.results_dir = results_dir
        self.monitor = monitor
        self.name = name

        self.sanity = False
        self.storage = {}

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module) -> None:
        self.ready = True

    def store(self, key, results) -> None:
        if key not in self.storage:
            self.storage[key] = self.init_logic()

        for k, v in results.items():
            if isinstance(v, (int, float, np.int, np.int32, np.int64, np.float)):
                self.storage[key][k].append(v)
            elif isinstance(v, list):
                self.storage[key][k] += v
            else:
                raise NotImplementedError

    def divide_data(self, outputs, batch) -> dict:
        def copy_batch(bt):
            new_bt = {}
            for k, v in bt.items():
                new_bt[k] = []
            return new_bt

        def copy_output(out):
            new_out = {}
            for k, v in out.items():
                new_out[k] = {}
                for k1, v1 in v.items():
                    if not isinstance(v1, dict):
                        new_out[k][k1] = []
                    else:
                        new_out[k][k1] = {}
                        for k2, v2 in v1.items():
                            new_out[k][k1][k2] = []
            return new_out

        def save_batch(gd, ag, bt, en):
            for k, v in bt.items():
                gd[ag][1][k].append(v[en])
            return gd

        def save_output(gd, ag, out, en):
            for k, v in out.items():
                for k1, v1 in v.items():
                    if not isinstance(v1, dict):
                        gd[ag][0][k][k1].append(v1[en])
                    else:
                        for k2, v2 in v1.items():
                            gd[ag][0][k][k1][k2].append(v2[en])
            return gd

        grouped_data = {"all": (outputs, batch)}
        for en, aug in enumerate(batch["augmentation"]):
            if aug not in grouped_data:
                grouped_data[aug] = (copy_output(outputs), copy_batch(batch))

            grouped_data = save_output(grouped_data, aug, outputs, en)
            grouped_data = save_batch(grouped_data, aug, batch, en)

        for k, _ in grouped_data.items():
            # batch post process
            for k1, v1 in grouped_data[k][1].items():
                if isinstance(batch[k1], torch.Tensor) and not isinstance(
                    v1, torch.Tensor
                ):
                    grouped_data[k][1][k1] = torch.stack(v1)

            # output post process
            for k1, v1 in grouped_data[k][0].items():
                for k2, v2 in grouped_data[k][0][k1].items():
                    if (
                        not isinstance(v2, dict)
                        and isinstance(outputs[k1][k2], torch.Tensor)
                        and not isinstance(v2, torch.Tensor)
                    ):
                        grouped_data[k][0][k1][k2] = torch.stack(v2)
                    elif isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            if isinstance(
                                outputs[k1][k2][k3], torch.Tensor
                            ) and not isinstance(v3, torch.Tensor):
                                grouped_data[k][0][k1][k2][k3] = torch.stack(v3)

        return grouped_data

    def init_logic(self) -> dict:
        raise NotImplementedError

    def batch_logic(self, outputs, batch) -> dict:
        raise NotImplementedError

    def end_logic(self, saved) -> dict:
        raise NotImplementedError

    def save_logic(self, monitor, trainer, result, extra) -> None:
        raise NotImplementedError

    def default_save_logic(self, monitor, trainer, result, extra) -> None:
        if trainer.logger:
            for key, val in result.items():
                trainer.logger.experiment.log_metrics({f"{key}/{monitor}": val})

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        grouped_data = self.divide_data(outputs, batch)
        for k, (out, bt) in grouped_data.items():
            result = self.batch_logic(out, bt)
            self.store(k, result)

    def on_test_epoch_end(self, trainer, pl_module):
        for k, saved in self.storage.items():
            result, extra = self.end_logic(saved)
            logging.info(
                f"The model performance on {self.name} for {k} subset of data is {result}."
            )
            self.default_save_logic(k, trainer, result, extra)
            self.save_logic(k, trainer, result, extra)

        self.storage = {}


class AccuracyMetric(MonitorBasedMetric):
    def __init__(self, monitor="all", name="acc", results_dir=""):
        super().__init__(monitor, name, results_dir)

    def init_logic(self) -> dict:
        return {"total": [], "correct": []}

    def batch_logic(self, outputs, batch):
        result = {
            "total": len(outputs["p2u_outputs"]["logits"]),
            "correct": np.sum(
                np.argmax(outputs["p2u_outputs"]["logits"].cpu().numpy(), axis=1)
                == outputs["p2u_outputs"]["p2u"]["labels"].cpu().numpy()
            ),
        }
        return result

    def end_logic(self, saved) -> dict:
        result = {"accuracy": sum(saved["correct"]) * 100 / sum(saved["total"])}
        extra = None
        return result, extra

    def save_logic(self, monitor, trainer, result, extra) -> None:
        pass


class CalibrationMetric(MonitorBasedMetric):
    def __init__(self, monitor="all", name="calibration", results_dir="", num_bins=10):
        super().__init__(monitor, name, results_dir)
        self.num_bins = num_bins

    def init_logic(self) -> dict:
        return {"y_prob_max": [], "correct": []}

    def batch_logic(self, outputs, batch) -> dict:
        result = {
            "y_prob_max": np.amax(
                outputs["p2u_outputs"]["logits"].softmax(dim=1).cpu().numpy(), axis=-1
            ).tolist(),
            "correct": (
                np.argmax(outputs["p2u_outputs"]["logits"].cpu().numpy(), axis=1)
                == outputs["p2u_outputs"]["p2u"]["labels"].cpu().numpy()
            )
            .astype(int)
            .tolist(),
        }
        return result

    def end_logic(self, saved) -> dict:
        bins = np.linspace(0.0, 1.0 + 1e-8, self.num_bins + 1)
        bin_ids = np.digitize(saved["y_prob_max"], bins) - 1

        bin_sums = np.bincount(
            bin_ids, weights=saved["y_prob_max"], minlength=len(bins)
        )
        bin_true = np.bincount(bin_ids, weights=saved["correct"], minlength=len(bins))
        bin_total = np.bincount(bin_ids, minlength=len(bins))

        non_zero = bin_total != 0
        prob_true = bin_true[non_zero] / bin_total[non_zero]
        prob_pred = bin_sums[non_zero] / bin_total[non_zero]

        expected_calibration_error = (
            np.sum(bin_total[non_zero] * np.abs(prob_true - prob_pred))
            / bin_total[non_zero].sum()
        )

        overconfidence_error = np.sum(
            bin_total[non_zero]
            * prob_pred
            * np.max(
                np.concatenate(
                    (
                        (prob_pred - prob_true).reshape(-1, 1),
                        np.zeros((1, len(prob_pred))).T,
                    ),
                    axis=1,
                ),
                axis=-1,
            )
            / bin_total[non_zero].sum()
        )

        result = {
            "expected_calibration_error": expected_calibration_error,
            "overconfidence_error": overconfidence_error,
        }

        extra = {"prob_pred": prob_pred, "prob_true": prob_true}

        return result, extra

    def save_logic(self, monitor, trainer, result, extra) -> None:
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        ax1.plot(extra["prob_pred"], extra["prob_true"], "s-", label="$dataset_name")

        ax1.set_xlabel("Confidence Probability")
        ax1.set_ylabel("Accuracy")
        ax1.legend(loc="lower right")
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_title("Calibration Plot")

        plt.tight_layout()

        if trainer.logger:
            plt.savefig(
                os.path.join(self.results_dir, f"calibration_{monitor}.png"),
                bbox_inches="tight",
            )
