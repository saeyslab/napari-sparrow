# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place

import hydra
import pyrootutils
from omegaconf import DictConfig

from napari_spongepy import utils

log = utils.get_pylogger(__name__)

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="segment.yaml"
)
def main(cfg: DictConfig) -> None:
    subset = cfg.subset
    if subset:
        subset = utils.parse_subset(subset)
        log.info(f"Subset is {subset}")
    # imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from napari_spongepy._segmentation_widget import _segmentation_worker

    if cfg.segmentation.get("method"):
        method = cfg.segmentation.method
    else:
        method = hydra.utils.instantiate(cfg.segmentation)

    worker = _segmentation_worker(
        cfg.dataset.image,
        method=method,
        subset=subset,
        # small chunks needed if subset is used
    )
    [s, _] = worker.work()
    output_path = cfg.paths.output_dir + "/segmentation.zarr"
    s.to_zarr(output_path)
    log.info(f"Segmentation saved to {output_path}")


if __name__ == "__main__":
    main()
