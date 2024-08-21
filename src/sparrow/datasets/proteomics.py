import os

import pooch
from spatialdata import SpatialData, read_zarr

from sparrow.datasets.registry import get_registry


def mibi_example() -> SpatialData:
    """Example proteomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("proteomics/mibi_tof/sdata_multi_channel.zarr.zip", processor=pooch.Unzip())
    return read_zarr(os.path.commonpath(unzip_path))
