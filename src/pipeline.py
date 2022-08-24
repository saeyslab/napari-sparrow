# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place


import hydra
import pyrootutils
from omegaconf import DictConfig

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="pipeline.yaml"
)
def main(cfg: DictConfig) -> None:
    from napari_spongepy import pipeline_functions as pf

    # The pipeline consist of 5 steps:
    # Clean
    cfg = pf.clean(cfg)

    # Segment
    cfg = pf.segment(cfg)

    # Allocate
    cfg = pf.allocate(cfg)

    # Annotate
    cfg = pf.annotate(cfg)

    # Visualize
    cfg = pf.visualize(cfg)

    return


if __name__ == "__main__":
    main()
