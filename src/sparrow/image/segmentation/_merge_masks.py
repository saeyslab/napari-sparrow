from __future__ import annotations

from typing import Optional, Tuple

import dask.array as da
import numpy as np
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image._image import _get_spatial_element
from sparrow.image.segmentation._apply import apply_labels_layers
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

from skimage.segmentation import relabel_sequential


def merge_labels_layers(
    sdata: SpatialData,
    labels_layer_1: str,
    labels_layer_2: str,
    depth: Tuple[int, ...] | int = 100,
    chunks: Optional[str | int | Tuple[int, ...]] = "auto",
    output_labels_layer: Optional[str] = None,
    output_shapes_layer: Optional[str] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
) -> SpatialData:
    sdata = apply_labels_layers(
        sdata,
        func=_merge_masks_block,
        labels_layers=[labels_layer_1, labels_layer_2],
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=True,
    )
    return sdata


def merge_labels_layers_nuclei(
    sdata: SpatialData,
    labels_layer: str,
    labels_layer_nuclei_expanded: str,
    labels_layer_nuclei: str,
    depth: Tuple[int, ...] | int = 100,
    chunks: Optional[str | int | Tuple[int, ...]] = "auto",
    output_labels_layer: Optional[str] = None,
    output_shapes_layer: Optional[str] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
) -> SpatialData:
    labels_layers = [labels_layer, labels_layer_nuclei_expanded, labels_layer_nuclei]
    for layer in labels_layers:
        if layer not in [*sdata.labels]:
            raise ValueError(
                f"Layer '{layer}' not found in available label layers '{[*sdata.labels]}' of sdata."
            )

    se_nuclei_expanded = _get_spatial_element(sdata, labels_layer_nuclei_expanded)
    se_nuclei = _get_spatial_element(sdata, labels_layer_nuclei)

    np.array_equal(
        da.unique(se_nuclei_expanded.data), da.unique(se_nuclei.data)
    ), f"Labels layer '{labels_layer_nuclei_expanded}' should contain same labels as '{labels_layer_nuclei}'."

    sdata = apply_labels_layers(
        sdata,
        func=_merge_masks_nuclei_block,
        labels_layers=labels_layers,
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=True,
    )
    return sdata


def _merge_masks_block(
    array_1: NDArray,  # array_1 gets priority
    array_2: NDArray,
) -> NDArray:
    # this is func for merging of arrays.
    # we need to relabel to avoid collisions in the merged_masks.
    array_1, _, _ = relabel_sequential(array_1)
    array_2, _, _ = relabel_sequential(array_2, offset=array_1.max() + 1)

    merged_masks = array_1
    merged_masks[merged_masks == 0] = array_2[merged_masks == 0]
    return merged_masks


def _merge_masks_nuclei_block(array_1, array_2, array_3):
    # array_1 is priority segmentation
    # array_2 is expanded_nucleus
    # array_3 is nucleus

    def _relabel_array(arr, original_values, new_values):
        relabeled_array = np.zeros_like(arr)
        assert original_values.shape == new_values.shape
        for new_label, old_label in zip(new_values, original_values):
            relabeled_array[arr == old_label] = new_label
        return relabeled_array

    # array_2 and array_3 can contain different labels due to chunking
    # (i.e. array_3 contains nuclei, which are smaller than expanded nuclei from array_2), but they need to be
    # relabeled in the same way.
    original_values = np.unique(
        np.concatenate([np.unique(array_2), np.unique(array_3)])
    )

    new_values = np.arange(original_values.size)
    array_2 = _relabel_array(
        array_2, original_values=original_values, new_values=new_values
    )
    array_3 = _relabel_array(
        array_3, original_values=original_values, new_values=new_values
    )

    # relabel array_1 to avoid collisions
    array_1, _, _ = relabel_sequential(
        array_1, offset=max(array_2.max(), array_3.max()) + 1
    )  # necessary, to avoid collisions

    unique_labels_3 = np.unique(
        array_3[array_3 > 0]
    )  # Get unique labels from array_3, excluding zero

    for label in unique_labels_3:
        # Determine the area of the label in array_3
        area_3 = np.sum(array_3 == label)

        # Calculate the overlap area of array_3's label with array_1
        overlap_area = np.sum((array_3 == label) & (array_1 > 0))

        # Check if more than half of the area is not in array_1
        if overlap_area <= area_3 / 2:
            # Find the corresponding area in array_2 and merge it into array_1
            merge_condition = (array_2 == label) & (array_1 == 0)
            array_1[merge_condition] = label

    return array_1
