from hydra.core.hydra_config import HydraConfig

from napari_spongepy.single import main


def test_pipeline(cfg_pipeline):
    """Run for 1 train, val and test step."""
    HydraConfig().set_config(cfg_pipeline)
    main(cfg_pipeline)
