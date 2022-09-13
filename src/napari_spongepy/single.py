# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place


import warnings
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


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

    # check the config
    check_config(cfg)

    # create a pure config without output paths that change every run
    full_cfg_dict = OmegaConf.to_object(cfg)
    pure_cfg_dict = full_cfg_dict.copy()
    pure_cfg_dict["paths"] = {}
    pure_cfg = OmegaConf.create(pure_cfg_dict)

    import cloudpickle
    import joblib

    from napari_spongepy import utils

    log = utils.get_pylogger(__name__)
    log.info(f"Pure config hash: {joblib.hash(cloudpickle.dumps(pure_cfg))}")

    from napari_spongepy import pipeline_functions as pf

    # Supress _core_genes futerewarnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    results: dict[str, Any] = {}

    dsk = {
        # The pipeline consist of 5 steps:
        "results_clean": (pf.clean, pure_cfg, results),
        "results_segment": (pf.segment, pure_cfg, "results_clean"),
        "results_allocate": (pf.allocate, pure_cfg, "results_segment"),
        "results_annotate": (pf.annotate, pure_cfg, "results_allocate"),
        "results_visualize": (pf.visualize, cfg, "results_annotate"),
    }

    if cfg.cache:
        import graphchain

        graphchain.get(dsk, "results_annotate")
    else:
        import dask

        dask.get(dsk, "results_annotate")


if __name__ == "__main__":
    main()
