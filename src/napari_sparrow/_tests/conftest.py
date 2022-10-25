import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


@pytest.fixture(scope="package")
def cfg_pipeline_global() -> DictConfig:
    # Expecting pytest to be run from the root dir. config_path should be relative to this file
    # The data_dir needs to be overwritten to point to the test data
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "dataset=resolve_liver",
                "segmentation=watershed",
                "dataset.image=${dataset.data_dir}/subset_20272_slide1_A1-1_DAPI.tiff",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)
    return cfg


# this is called by each test which uses `cfg_pipeline` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_pipeline(cfg_pipeline_global, tmp_path) -> DictConfig:
    cfg = cfg_pipeline_global.copy()
    yield cfg

    GlobalHydra.instance().clear()
