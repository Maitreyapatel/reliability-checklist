from typing import Any, List

import torch
from pytorch_lightning import LightningModule


class InferenceLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        pass

    def step(self, batch: Any):
        x, y = self.net.input2uniform(batch)
        outputs = self.forward(x)
        return outputs, y

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        outputs, targets = self.step(batch)
        return {"p2u_outputs": self.net.prediction2uniform(outputs), "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "inference_module.yaml")
    _ = hydra.utils.instantiate(cfg)
