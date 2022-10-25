"""" This file acts as a starting point for running the pipeline for multi sample analysis.
It can be run from any place with the command: 'sparrow-multi'. """


import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="configs", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    """Main function for the multi pipeline."""

    print(cfg)
    print(cfg.dataset)

    return


if __name__ == "__main__":
    main()
