# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place


import warnings
from typing import Any

import hydra
from omegaconf import DictConfig


def check_config(cfg: DictConfig):
    from pathlib import Path

    # check all mandatory paths
    for p in [
        cfg.paths.data_dir,
        cfg.dataset.data_dir,
        cfg.dataset.image,
        cfg.dataset.coords,
        cfg.paths.output_dir,
    ]:
        assert Path(p).exists()


@hydra.main(version_base="1.2", config_path="configs", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    from napari_spongepy import pipeline_functions as pf

    # check the config
    check_config(cfg)

    # Supress _core_genes futerewarnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # The pipeline consist of 5 steps:
    results: dict[str, Any] = {}
    # Clean
    cfg, results = pf.clean(cfg, results)

    # Segment
    cfg, results = pf.segment(cfg, results)

    # Allocate
    cfg, results = pf.allocate(cfg, results)

    # Annotate
    cfg, results = pf.annotate(cfg, results)

    # Visualize
    cfg, results = pf.visualize(cfg, results)

    return


if __name__ == "__main__":
    main()
