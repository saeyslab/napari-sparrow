"""" This file acts as a starting point for running the pipeline for single sample analysis.
It can be run from any place with the command: 'sparrow'. """

import logging
import os
import warnings
from typing import Any

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def check_config(cfg: DictConfig):
    """Checks if all paths and dataset paths are existing files, raise assertionError if not."""
    from pathlib import Path

    # Check if all mandatory paths exist
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
    """Main function for the single pipeline which checks the supplied paths first and then calls all five steps from the pipeline functions."""
    from napari_sparrow import pipeline_functions as pf

    # Checks the config paths, see the src/napari_sparrow/configs and local configs folder for settings
    check_config(cfg)

    # Supress _core_genes futerewarnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # The actual pipeline which consists of 7 steps:

    # Load
    log.info("Converting to zarr and (lazy) loading of SpatialData object.")
    sdata = pf.load(cfg)

    # Clean
    sdata = pf.clean(cfg, sdata)

    # Segment
    sdata = pf.segment(cfg, sdata)

    # Allocate
    sdata = pf.allocate(cfg, sdata)

    # Annotate
    sdata, mg_dict = pf.annotate(cfg, sdata)

    # Visualize
    sdata = pf.visualize(cfg, sdata, mg_dict)

    return


if __name__ == "__main__":
    main()
