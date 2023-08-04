"""" This file acts as a starting point for running the pipeline for single sample analysis.
It can be run from any place with the command: 'sparrow'. """

import logging
import glob
import warnings

import hydra
from omegaconf import DictConfig, ListConfig

log = logging.getLogger(__name__)


def check_config(cfg: DictConfig):
    """Checks if all paths and dataset paths are existing files, raise assertionError if not."""
    from pathlib import Path

    # Define the paths to check
    paths_to_check = [
        cfg.paths.data_dir,
        cfg.dataset.data_dir,
        cfg.dataset.coords,
        cfg.paths.output_dir,
    ]

    # If cfg.dataset.image is a list of paths, extend the paths_to_check with this list
    if isinstance(cfg.dataset.image, ListConfig):
        paths_to_check.extend(cfg.dataset.image)
    # Otherwise, just add the single path to paths_to_check
    else:
        paths_to_check.append(cfg.dataset.image)

    # Check if all mandatory paths exist
    for p in paths_to_check:
        # Check if the path contains a wildcard
        if '*' in p:
            matches = glob.glob(p)
            # Assert that at least one file matching the glob pattern exists
            assert matches, f"No file matches the path pattern {p}"
        else:
            assert Path(p).exists(), f"Path {p} does not exist."


@hydra.main(version_base="1.2", config_path="configs", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    """Main function for the single pipeline which checks the supplied paths first and then calls all five steps from the pipeline functions."""
    from napari_sparrow import pipeline as nasp

    # Checks the config paths, see the src/napari_sparrow/configs and local configs folder for settings
    check_config(cfg)

    # Supress _core_genes futerewarnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # The actual pipeline which consists of 7 steps:

    # Load
    log.info("Converting to zarr and (lazy) loading of SpatialData object.")
    sdata = nasp.load(cfg)

    # Clean
    sdata = nasp.clean(cfg, sdata)

    # Segment
    sdata = nasp.segment(cfg, sdata)

    # Allocate
    sdata = nasp.allocate(cfg, sdata)

    # Annotate
    sdata, mg_dict = nasp.annotate(cfg, sdata)

    # Visualize
    sdata = nasp.visualize(cfg, sdata, mg_dict)

    return


if __name__ == "__main__":
    main()
