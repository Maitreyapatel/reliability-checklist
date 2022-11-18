import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_eval_config(cfg_eval: DictConfig):
    assert cfg_eval
    assert cfg_eval.datamodule
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.datamodule)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)
