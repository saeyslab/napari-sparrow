from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import pandas as pd
from dask.array import unique
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.segmentation import relabel_sequential
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image._image import _get_spatial_element
from sparrow.image.segmentation._apply import apply_labels_layers
from sparrow.image.segmentation._utils import _SEG_DTYPE, _rechunk_overlap
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


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
    # TODO: do this more strict. Check if more than half of label in array_2 is not in
    # one of the labels in array_1
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


def mask_to_original(
    sdata: SpatialData,
    labels_layer: str,
    original_labels_layers: List[str],
    depth: Tuple[int, ...] | int = 400,
    chunks: str | int | Tuple[int, ...] = "auto",
) -> DataFrame:
    """
    Maps labels from a mask layer to their corresponding labels in original labels layers within a SpatialData object.
    The labels in `labels_layers` will be mapped to the label in of the the labels layers in `original_labels_layers` 
    with which it has maximum overlap.  

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the mask and original labels layers.
    labels_layer : str
        The name of the labels layer used as a mask for mapping.
    original_labels_layers : List[str]
        The names of the original labels layers to which the mask labels are mapped.
    depth : Tuple[int, ...] | int, default=400
        The depth around the boundary of each block to load when the array is split into blocks. This ensures
        that the split doesn't cause misalignment along the edges. Default is 400. Set depth larger than the maximum
        cell size to avoid chunking effects.
    chunks : str | int | Tuple[int, ...], default="auto"
        Specification for rechunking the data before applying the function. If chunks is a Tuple, they should contain
        desired chunk size for 'y', 'x'. 'auto' allows the function to determine optimal chunking.

    Returns
    -------
    DataFrame
        A pandas DataFrame where each row corresponds to a unique cell id from the mask layer, and columns correspond
        to the original labels layers. Each cell in the DataFrame contains the label from the original layer that
        overlaps most with the mask label.

    Raises
    ------
    AssertionError
        - If arrays from different labels layers do not have the same shape.
        - If depth is provided as a Tuple but does not match (y, x) dimensions.
        - If chunks is a Tuple, and does not match (y, x) dimensions.
        - If the number of blocks in the z-dimension is not equal to 1.

    Notes
    -----
    This function is designed to facilitate the comparison or integration of segmentation results by mapping mask
    labels back to their original labels. It handles arrays with different dimensions and ensures that chunking
    and depth parameters are appropriately applied for efficient computation.
    """
    labels_arrays = [sdata.labels[labels_layer].data]

    cell_ids = unique(labels_arrays[0]).compute()

    for _labels_layer in original_labels_layers:
        labels_arrays.append(sdata.labels[_labels_layer].data)

    # Check for consistent shapes
    first_shape = labels_arrays[0].shape
    for x_label in labels_arrays:
        assert (
            x_label.shape == first_shape
        ), "Only arrays with same shape are currently supported."

    # First make dimension uniform (z,y,x).
    _labels_arrays = []
    for i, x_label in enumerate(labels_arrays):
        if x_label.ndim == 2:
            _labels_arrays.append(x_label[None, ...])
        else:
            _labels_arrays.append(x_label)

    _x_label = _labels_arrays[0]

    if isinstance(depth, int):
        depth = {0: 0, 1: depth, 2: depth}
    else:
        assert (
            len(depth) == _x_label.ndim - 1
        ), "Please (only) provide depth for ( 'y', 'x')."
        # set depth for every dimension
        depth2 = {0: 0, 1: depth[0], 2: depth[1]}
        depth = depth2

    if chunks is not None:
        if not isinstance(chunks, (int, str)):
            assert (
                len(chunks) == _x_label.ndim - 1
            ), "Please (only) provide chunks for ( 'y', 'x')."
            chunks = (_x_label.shape[0], chunks[0], chunks[1])

    rechunked_arrays = []
    for i, x_label in enumerate(_labels_arrays):
        #  rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
        x_label = _rechunk_overlap(x_label, depth=depth, chunks=chunks)
        assert (
            x_label.numblocks[0] == 1
        ), f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label.numblocks[0]}`."
        rechunked_arrays.append(x_label)

    def _mask_to_original_chunks(
        x_labels_gs: NDArray,
        *arrays: Tuple[NDArray],
        block_info,
        _depth: Dict[int, int],
    ):
        def _zero_non_max(list_1, list_2):
            if not list_1 or not list_2 or len(list_1) != len(list_2):
                raise ValueError("Lists should be non-empty and of the same length.")
            max_value = max(list_1)
            for i in range(len(list_1)):
                if list_1[i] != max_value:
                    list_2[i] = 0
            return list_2

        total_blocks = block_info[0]["num-chunks"]
        assert (
            total_blocks[0] == 1
        ), "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
        assert (
            depth[0] == 0
        ), "Depth not equal to 0 in z dimension is currently not supported."
        assert len(depth) == 3, "Please provide depth values for z,y and x."

        # img_gs, gold standard labels
        labels_gs = np.unique(
            x_labels_gs[:, _depth[1] : -_depth[1], _depth[2] : -_depth[2]]
        )

        result = np.zeros((cell_ids.shape[0], len(arrays)), dtype=_SEG_DTYPE)

        for label in labels_gs:
            if label == 0:
                continue
            max_label_list = []
            max_area_list = []

            for _array in arrays:
                positions = np.where(x_labels_gs == label)
                overlapping_labels = _array[positions]

                label_areas = {
                    lbl: np.sum(overlapping_labels == lbl)
                    for lbl in np.unique(overlapping_labels)
                }
                label_areas.pop(0, None)

                # Find the label with the maximum area
                if label_areas:
                    max_label = max(label_areas, key=label_areas.get)
                    max_area = label_areas[max_label]
                else:
                    max_label = 0  # Set to 0 if there's no overlap
                    max_area = 0

                max_label_list.append(max_label)
                max_area_list.append(max_area)

            max_overlap = _zero_non_max(max_area_list, max_label_list)

            index = np.where(cell_ids == label)[0][0]
            result[index] = max_overlap

        return result

    result = da.map_overlap(
        lambda *arrays, block_info=None, _depth=depth: _mask_to_original_chunks(
            *arrays, block_info=block_info, _depth=_depth
        ),  # Unpack and pass all arrays to _mask_to_original_chunks
        *rechunked_arrays,  # Unpack the list of Dask arrays as individual arguments
        dtype=_SEG_DTYPE,
        drop_axis=0,
        depth=depth,
        trim=False,
        boundary=0,
        chunks=(len(cell_ids), len(original_labels_layers)),
    )

    result_of_chunks = da.zeros(
        (len(cell_ids), len(original_labels_layers)), dtype=result.dtype
    )

    num_chunks = result.numblocks

    # sum the result for each chunk
    for i in range(num_chunks[0]):
        for j in range(num_chunks[1]):
            current_chunk = result.blocks[i, j]
            condition = result_of_chunks == 0
            result_of_chunks = da.where(
                condition,
                current_chunk,  # if equal to zero, overwrite, else, keep old value
                result_of_chunks,
            )

    result_computed = result_of_chunks.compute()
    df = pd.DataFrame(
        result_computed,
        index=cell_ids,
        columns=original_labels_layers,
    )

    df.drop(0, inplace=True, axis=0)
    df.index = df.index.astype(str)

    return df
