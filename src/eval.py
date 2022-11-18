import hydra
import pyrootutils
from omegaconf import DictConfig, open_dict

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    if "debugme" in cfg:
        import debugpy

        strport = cfg.debugme
        debugpy.listen(strport)
        print(
            f"waiting for debugger on {strport}. Add the following to your launch.json and start the VSCode debugger with it:"
        )
        print(
            f'{{\n    "name": "Python: Attach",\n    "type": "python",\n    "request": "attach",\n    "connect": {{\n      "host": "localhost",\n      "port": {strport}\n    }}\n }}'
        )
        debugpy.wait_for_client()

        with open_dict(cfg):
            cfg.trainer.gpus = 0
            # cfg.trainer.accelerator = None
            cfg.trainer.strategy = None
            cfg.loggers = {}

    from src.tasks.eval_task import evaluate

    evaluate(cfg)


if __name__ == "__main__":
    main()
