from .registry import get_ome_registry, get_registry, get_spatialdata_registry
from .transcriptomics import (
    merscope_example,
    merscope_segmentation_masks_example,
    resolve_example,
    resolve_example_multiple_coordinate_systems,
    visium_hd_example,
    visium_hd_example_custom_binning,
    xenium_example,
)

__all__ = [
    "get_ome_registry",
    "get_registry",
    "get_spatialdata_registry",
    "merscope_example",
    "merscope_segmentation_masks_example",
    "resolve_example",
    "resolve_example_multiple_coordinate_systems",
    "visium_hd_example",
    "visium_hd_example_custom_binning",
    "xenium_example",
]
