# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place


from typing import Any

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.2", config_path="configs", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    from napari_spongepy import pipeline_functions as pf

    # The pipeline consist of 5 steps:
    results: dict[str, Any] = {}
    # Clean
    cfg, results = pf.clean(cfg, results)

    # Segment
    # cfg, results = pf.segment(cfg, results)

    # Allocate
    # cfg, results = pf.allocate(cfg, results)

    # Annotate
    # cfg, results = pf.annotate(cfg, results)

    # Visualize
    # cfg, results = pf.visualize(cfg, results)

    return


if __name__ == "__main__":
    main()
