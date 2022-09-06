# this file acts as a robust starting point for launching hydra multiruns
# can be run from any place


import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="configs", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    pass

    print(cfg)
    print(cfg.dataset)

    return


if __name__ == "__main__":
    main()
