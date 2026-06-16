"""This file tests the entire pipeline and should be used for development purposes."""

import importlib.util

import pytest
from hydra.core.hydra_config import HydraConfig

from sparrow.single import main


# Skip the full pipeline test when any of its required optional libraries are absent.
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose")
    or not importlib.util.find_spec("basicpy")
    or not importlib.util.find_spec("squidpy"),
    reason="requires the cellpose, basicpy and squidpy libraries",
)
def test_pipeline(cfg_pipeline):
    HydraConfig().set_config(cfg_pipeline)
    main(cfg_pipeline)
