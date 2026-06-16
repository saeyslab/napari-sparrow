import os

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from spatialdata import read_zarr
from spatialdata.datasets import blobs

from sparrow.datasets.registry import get_registry
from sparrow.datasets.transcriptomics import (
    resolve_example,
    resolve_example_multiple_coordinate_systems,
    visium_hd_example_custom_binning,
)

try:
    from sparrow.datasets.pixie_example import pixie_example
except ImportError:
    pixie_example = None  # type: ignore[assignment]

try:
    from sparrow.datasets.proteomics import mibi_example
except ImportError:
    mibi_example = None  # type: ignore[assignment]


@pytest.fixture(scope="function")
def cfg_pipeline_global(path_dataset_markers) -> DictConfig:
    # Expecting pytest to be run from the root dir. config_path should be relative to this file
    # The data_dir needs to be overwritten to point to the test data

    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    registry = get_registry()
    dataset_image = registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_DAPI_4288_2144.tiff")
    dataset_coords = registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_results_4288_2144.txt")

    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                f"paths.data_dir={root}",
                f"dataset.data_dir={root}",
                f"dataset.image={dataset_image}",
                f"dataset.coords={dataset_coords}",
                f"dataset.markers={path_dataset_markers}",
                "allocate.delimiter='\t'",
                "allocate.column_x=0",
                "allocate.column_y=1",
                "allocate.column_gene=3",
                "segmentation=cellpose",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

    return cfg


# this is called by each test which uses `cfg_pipeline` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_pipeline(cfg_pipeline_global, tmp_path):
    cfg = cfg_pipeline_global.copy()

    cfg.paths.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture
def sdata_multi_c(tmpdir):
    if mibi_example is None:
        pytest.skip("sparrow.datasets.proteomics not available")
    sdata = mibi_example()
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata.zarr"))
    yield sdata


@pytest.fixture
def sdata_multi_c_no_backed():
    if mibi_example is None:
        pytest.skip("sparrow.datasets.proteomics not available")
    sdata = mibi_example()
    yield sdata


@pytest.fixture
def sdata_transcripts(tmpdir):
    sdata = resolve_example()
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    yield sdata


@pytest.fixture
def sdata_transcripts_no_backed():
    sdata = resolve_example()
    yield sdata


@pytest.fixture
def sdata_transcripts_mul_coord(tmpdir):
    sdata = resolve_example_multiple_coordinate_systems()
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    yield sdata


@pytest.fixture
def sdata_bin():
    sdata = visium_hd_example_custom_binning()
    yield sdata


@pytest.fixture
def sdata():
    yield blobs(length=1000, n_channels=3)


@pytest.fixture
def sdata_blobs():
    # Define the channel names used by tests that rely on blob image fixtures.
    # These names must match the c_coords passed to blobs() so that .sel(c=channel_name) works.
    c_coords = [
        "nucleus",
        "lineage_0",
        "lineage_1",
        "lineage_2",
        "lineage_3",
        "lineage_4",
        "lineage_5",
        "lineage_6",
        "lineage_7",
        "lineage_8",
        "lineage_9",
    ]
    sdata = blobs(length=64, n_channels=len(c_coords), c_coords=c_coords)
    # blobs() stores table var_names as "channel_{name}_sum" by default, but tests and
    # calculate_snr_ratio look up channels by bare name (e.g. "nucleus").
    # Renaming the index makes table.var_names consistent with the image channel names.
    sdata["table"].var.index = c_coords
    yield sdata


@pytest.fixture
def sdata_pixie():
    if pixie_example is None:
        pytest.skip("sparrow.datasets.pixie_example not available")
    sdata = pixie_example()
    yield sdata


@pytest.fixture
def path_dataset_markers():
    registry = get_registry()
    return registry.fetch("transcriptomics/resolve/mouse/dummy_markers.csv")
