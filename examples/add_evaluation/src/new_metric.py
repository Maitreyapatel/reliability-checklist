import numpy as np
from sklearn.metrics import f1_score

from reliability_checklist.callbacks.evals.discriminative import MonitorBasedMetric


class F1Metric(MonitorBasedMetric):
    def __init__(
        self,
        monitor="all",
        name="f1score",
        results_dir="",
        override=None,
        radar=True,
        max_possible=1.0,
        inverse=False,
        average=None,
    ):
        super().__init__(monitor, name, results_dir, override, radar, max_possible, inverse)
        self.average = average

    def init_logic(self) -> dict:
        return {"y_pred": [], "y_true": []}

    def batch_logic(self, outputs, batch):
        result = {
            "y_true": outputs["p2u_outputs"]["p2u"]["labels"].cpu().numpy(),
            "y_pred": np.argmax(outputs["p2u_outputs"]["logits"].cpu().numpy(), axis=1),
        }
        return result

    def end_logic(self, saved) -> dict:
        result = {"f1score": f1_score(saved["y_true"], saved["y_pred"], average=self.average)}
        extra = None
        return result, extra

    def save_logic(self, monitor, trainer, result, extra) -> None:
        pass
