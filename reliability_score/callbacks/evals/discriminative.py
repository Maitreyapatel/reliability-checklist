import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from sklearn import metrics


class MonitorBasedMetric(Callback):
    def __init__(
        self, monitor, name, results_dir, override, radar, max_possible, inverse
    ) -> None:
        self.results_dir = results_dir
        self.override = override
        self.monitor = monitor
        self.name = name
        self.radar = radar
        self.max_possible = max_possible
        self.inverse = inverse

        if not os.path.exists(self.results_dir) and "None" not in self.results_dir:
            os.mkdir(self.results_dir)

        self.max_val = max_possible
        self.min_val = (
            0  # Always assumed to be zero. But this might change based on the metric.
        )
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
            if isinstance(v, (int, float, np.int32, np.int64)):
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
        if self.override == "mixed":
            return grouped_data

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

    def get_max_possible_score(self) -> None:
        raise NotImplementedError

    def default_save_logic(self, monitor, trainer, result, extra) -> None:
        if trainer.logger:
            for key, val in result.items():
                trainer.logger.experiment.log({f"{key}/{monitor}": val})

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        grouped_data = self.divide_data(outputs, batch)
        for k, (out, bt) in grouped_data.items():
            result = self.batch_logic(out, bt)
            self.store(k, result)

    def get_scaled_values(self, x):
        assert self.max_val != None
        if self.inverse:
            return 100 - (100 * x / self.max_val)
        else:
            return 100 * x / self.max_val

    def create_radar(self, trainer, results):
        radar_set = {}
        tmp_ = {
            "subjects": [],
            "model1": [],
        }
        for k, v in results.items():
            if k == "all":
                continue

            for k1, v1 in v.items():
                if k1 not in radar_set:
                    radar_set[k1] = deepcopy(tmp_)
                radar_set[k1]["subjects"].append(k)
                radar_set[k1]["model1"].append(self.get_scaled_values(v1))

        for metric_name, _ in radar_set.items():
            angles = np.linspace(
                0, 2 * np.pi, len(radar_set[metric_name]["subjects"]), endpoint=False
            )
            angles = np.concatenate((angles, [angles[0]]))
            radar_set[metric_name]["subjects"].append(radar_set[metric_name]["subjects"][0])
            radar_set[metric_name]["model1"].append(radar_set[metric_name]["model1"][0])

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(polar=True)  # basic plot
            ax.plot(
                angles,
                radar_set[metric_name]["model1"],
                "o--",
                color="g",
                label="model1",
            )
            # fill plot
            ax.fill(angles, radar_set[metric_name]["model1"], alpha=0.25, color="g")
            # Add labels
            ax.set_thetagrids(angles * 180 / np.pi, radar_set[metric_name]["subjects"])
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.ylim(0, 100)
            plt.title(metric_name)

            if trainer.logger:
                plt.savefig(
                    os.path.join(self.results_dir, f"radar_{metric_name}.png"),
                    bbox_inches="tight",
                )
            else:
                logging.error(
                    "Could not save the radar chart as the logger is missing."
                )

    def on_test_epoch_end(self, trainer, pl_module):
        saved_results = {}
        for k, saved in self.storage.items():
            result, extra = self.end_logic(saved)
            logging.info(
                f"The model performance on {self.name} for {k} subset of data is {result}."
            )
            self.default_save_logic(k, trainer, result, extra)
            self.save_logic(k, trainer, result, extra)
            saved_results[k] = result

        if self.radar:
            self.create_radar(trainer, saved_results)
        self.storage = {}


class AccuracyMetric(MonitorBasedMetric):
    def __init__(
        self,
        monitor="all",
        name="acc",
        results_dir="",
        override=None,
        radar=True,
        max_possible=100,
        inverse=False,
    ):
        super().__init__(
            monitor, name, results_dir, override, radar, max_possible, inverse
        )

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
    def __init__(
        self,
        monitor="all",
        name="calibration",
        results_dir="",
        num_bins=10,
        override=None,
        radar=True,
        max_possible=0.5,
        inverse=False,
    ):
        super().__init__(
            monitor, name, results_dir, override, radar, max_possible, inverse
        )
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


class SensitivityMetric(MonitorBasedMetric):
    def __init__(
        self,
        monitor="all",
        name="sensitivity",
        results_dir="",
        override="mixed",
        radar=True,
        max_possible="dynamic",
        inverse=False,
    ):
        super().__init__(
            monitor, name, results_dir, override, radar, max_possible, inverse
        )
        self.default_mapping = {}

    def init_logic(self) -> dict:
        return {"map": [], "logits": [], "aug_pred": [], "aug_true": []}

    def get_max_possible_score(self, num_classes):
        self.max_val = self.entropy(torch.ones(num_classes) * 0.5, dim=0)

    def batch_logic(self, outputs, batch):
        result = self.init_logic()
        if self.max_val == None:
            self.get_max_possible_score(outputs["p2u_outputs"]["logits"].shape[1])

        for i in range(len(outputs["p2u_outputs"]["logits"])):
            if batch["augmentation"][i] == "DEFAULT":
                tmp = {"logits": [], "y_pred": [], "y_true": []}
                tmp["logits"].append(outputs["p2u_outputs"]["logits"][i].cpu())
                tmp["y_pred"].append(
                    torch.argmax(outputs["p2u_outputs"]["logits"][i]).cpu().data.numpy()
                )
                tmp["y_true"].append(batch["label"][i].cpu().data.numpy())
                self.default_mapping[
                    int(batch["primary_key"][i].cpu().data.numpy())
                ] = tmp

            if batch["augmentation"][i] == "parrot":
                result["map"].append(batch["mapping"][i].cpu().data.numpy())
                result["logits"].append(outputs["p2u_outputs"]["logits"][i].cpu())
                result["aug_pred"].append(
                    torch.argmax(outputs["p2u_outputs"]["logits"][i]).cpu().data.numpy()
                )
                result["aug_true"].append(batch["label"][i].cpu().data.numpy())

        return result

    def entropy(self, outputs, dim=1):
        outputs = torch.nn.functional.softmax(outputs, dim=dim)
        en = (torch.log(outputs) * outputs).sum(dim=dim) * -1
        return en

    def end_logic(self, saved) -> dict:
        extra = None
        if saved["logits"] == []:
            logging.error(
                "Attempted to use sensitivity metric without parrot augmentations."
            )
            return {"sensitivity": -1}, extra

        saved["entropy"] = self.entropy(torch.stack(saved["logits"]), dim=1)
        en_max = self.entropy(
            torch.tensor([0.5 for i in range(len(saved["logits"][0]))]), dim=0
        )

        overall_sensitivity = []
        sensitivity_dict = {}
        for i in range(len(saved["map"])):
            if int(saved["map"][i]) not in sensitivity_dict:
                sensitivity_dict[int(saved["map"][i])] = []
            org_data = self.default_mapping[int(saved["map"][i])]
            en_org = self.entropy(org_data["logits"][0], dim=0)
            en_aug = saved["entropy"][i]
            sense = abs(float(en_org) - float(en_aug)) + 2 * (
                int(org_data["y_true"][0] == saved["aug_true"][i])
                * int(org_data["y_pred"][0] != saved["aug_pred"][i])
                + int(org_data["y_true"][0] != saved["aug_true"][i])
                * int(org_data["y_pred"][0] == saved["aug_pred"][i])
            ) * (float(en_max) - float(max(en_org, en_aug)))
            sensitivity_dict[int(saved["map"][i])].append(sense)
        overall_sensitivity = [np.mean(v) for k, v in sensitivity_dict.items()]
        result = {"sensitivity": np.mean(overall_sensitivity)}

        return result, extra

    def save_logic(self, monitor, trainer, result, extra) -> None:
        pass


class SelectivePredictionMetric(MonitorBasedMetric):
    def __init__(
        self,
        monitor="all",
        name="selective_prediction",
        results_dir="",
        override=None,
        radar=True,
        max_possible=1,
        inverse=False,
    ):
        super().__init__(
            monitor, name, results_dir, override, radar, max_possible, inverse
        )

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

        tuples = [
            (probs, correct)
            for probs, correct, in zip(saved["y_prob_max"], saved["correct"])
        ]
        sorted_tuples = sorted(tuples, key=lambda x: -x[0])
        sorted_probs = [x[0] for x in sorted_tuples]
        sorted_em = [x[1] for x in sorted_tuples]
        total_questions = len(sorted_em)
        total_correct = 0
        covered = 0
        risks = []
        coverages = []

        for em, prob in zip(sorted_em, sorted_probs):
            covered += 1
            if em:
                total_correct += 1
            risks.append(1 - (total_correct / covered))
            coverages.append(covered / total_questions)

        auc = round(metrics.auc(coverages, risks), 4)

        result = {"auc_selective_prediction": auc}

        extra = {"coverage": coverages, "risk": risks}
        return result, extra

    def save_logic(self, monitor, trainer, result, extra) -> None:

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        ax1.plot(extra["coverage"], extra["risk"], label="$dataset_name")

        ax1.set_xlabel("Coverage")
        ax1.set_ylabel("Risk")
        ax1.legend(loc="lower right")
        ax1.set_title("Selective Prediction Plot")

        plt.tight_layout()

        if trainer.logger:
            plt.savefig(
                os.path.join(self.results_dir, f"selective_prediction_{monitor}.png"),
                bbox_inches="tight",
            )
