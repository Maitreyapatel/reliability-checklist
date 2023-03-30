import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from reliability_checklist.tasks import eval_task


@pytest.mark.slow
def test_eval(cfg_eval):
    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = eval_task.evaluate(cfg_eval)
