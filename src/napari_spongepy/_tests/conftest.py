import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


@pytest.fixture(scope="package")
def cfg_pipeline_global() -> DictConfig:
    with initialize(version_base="1.2", config_path="../../../configs"):
        cfg = compose(config_name="pipeline", overrides=["+dataset=resolve"])

    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_pipeline(cfg_pipeline_global, tmp_path) -> DictConfig:
    cfg = cfg_pipeline_global.copy()
    yield cfg

    GlobalHydra.instance().clear()
