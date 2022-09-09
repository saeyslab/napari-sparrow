import pytest


@pytest.mark.skip(
    reason="Hydra is broken on CI: hydra.errors.MissingConfigException: In 'pipeline': Could not find 'viz/default'"
)
def test_pipeline(cfg_pipeline):
    """Run for 1 train, val and test step."""
    # HydraConfig().set_config(cfg_pipeline)
    # main(cfg_pipeline)
