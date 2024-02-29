"""This file tests the entire pipeline and should be used for development purposes."""
from hydra.core.hydra_config import HydraConfig

from sparrow.single import main


def test_pipeline(cfg_pipeline):
    HydraConfig().set_config(cfg_pipeline)
    main(cfg_pipeline)
