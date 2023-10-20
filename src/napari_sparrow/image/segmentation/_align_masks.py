from __future__ import annotations

from typing import Any, List, Optional, Tuple

import dask.array as da
import numpy as np
from dask.array import Array
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from napari_sparrow.image._image import (
    _add_label_layer,
    _get_spatial_element,
    _get_translation,
)
from napari_sparrow.image.segmentation._utils import (
    _SEG_DTYPE,
    _add_depth_to_chunks_size,
    _check_boundary,
    _clean_up_masks,
    _trim_masks,
    _rechunk_overlap,
)
from napari_sparrow.shape._shape import _add_shapes_layer
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def align_labels_layers(
    sdata: SpatialData,
    labels_layer_1: str,
    labels_layer_2: str,
    depth: Tuple[int, int] = (100, 100),
    chunks: Optional[str | int | tuple[int, ...]] = "auto",
    output_labels_layer: Optional[str] = None,
    output_shapes_layer: Optional[str] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
):
    """
    This function aligns two label layers by examining the labels in labels_layer_1
    and identifying their maximum overlap with labels in labels_layer_2.
    It then updates the labels in labels_layer_1,
    reassigning them to match the corresponding overlapping label values from labels_layer_2.
    The function can also generate a shapes layer based on the resulting output_labels_layer.
    The layers are identified by their names and must exist within the SpatialData object passed.
    Usually, labels_layer_1 consists of masks derived from nucleus segmentation,
    while labels_layer_2 contains masks resulting from whole cell segmentation.

    Parameters
    ----------
    sdata : SpatialData
        The spatial data object containing the labels layers to be aligned.
    labels_layer_1 : str
        The name of the first labels layer to align.
    labels_layer_2 : str
        The name of the second labels layer to align.
    depth : Tuple[int, int], optional
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is (100, 100). Please set depth>cell size to avoid chunking effects.
    chunks : Optional[str | int | tuple[int, ...]], optional
        The desired chunk size for the Dask computation, or "auto" to allow the function to
        choose an optimal chunk size based on the data. Default is "auto".
    output_labels_layer : Optional[str], optional
        The name for the new labels layer generated after alignment. If None and overwrite is False,
        a ValueError is raised. If None and overwrite is True, 'labels_layer_1' will be overwritten
        with the aligned layer. Default is None.
    output_shapes_layer : Optional[str], optional
        The name for the new shapes layer generated from the aligned labels layer. If None, no shapes
        layer is created. Default is None.
    scale_factors : Optional[ScaleFactors_t], optional
        Scale factors to apply for multiscale.
    overwrite : bool, optional
        If True, allows the function to overwrite the data in 'output_labels_layer' with the aligned
        data. If False and 'output_labels_layer' is None, a ValueError is raised. Default is False.

    Returns
    -------
    SpatialData
        The modified spatial data object with the aligned labels layers and potentially new layers
        based on the alignment.

    Raises
    ------
    AssertionError
        If the shapes of the label arrays or their translations do not match.
    ValueError
        If 'output_labels_layer' is None and 'overwrite' is False, indicating ambiguity in the
        user's intent.

    Notes
    -----
    - The function works with Dask arrays and can handle large datasets that don't fit into memory.
    - Only arrays with the same shape are supported for alignment. Misaligned arrays could be due to
      various reasons, including errors in previous processing steps or physical shifts in the samples.
    - The alignment respects the original labelling but ensures that corresponding areas in both layers
      match after the process.

    Examples
    --------
    >>> sdata = align_labels_layers(sdata, 'layer_1', 'layer_2', depth=(50, 50), overwrite=True)
    >>> # 'sdata' now has aligned 'layer_1' and 'layer_2', potentially with new layers based on the alignment.
    """

    se_1 = _get_spatial_element(sdata, layer=labels_layer_1)
    se_2 = _get_spatial_element(sdata, layer=labels_layer_2)

    x_label_1 = se_1.data
    x_label_2 = se_2.data

    assert (
        x_label_1.shape == x_label_2.shape
    ), "Only arrays with same shape are currently supported, "
    f"but labels layer with name {labels_layer_1} has shape {x_label_1.shape}, "
    f"while labels layer with name {labels_layer_2} has shape {x_label_2.shape}  "

    t1x, t1y = _get_translation(se_1)
    t2x, t2y = _get_translation(se_2)

    assert (t1x, t1y) == (
        t2x,
        t2y,
    ), f"labels layer 1 with name {labels_layer_1} should "
    f"have same translation as labels layer 1 with name {labels_layer_2}"

    x_label_aligned = _align_dask_arrays(
        x_label_1, x_label_2, chunks=chunks, depth=depth
    )

    if output_labels_layer is None:
        output_labels_layer = labels_layer_1
        if overwrite == False:
            raise ValueError(
                "output_labels_layer was set to None, but overwrite to False. "
                f"to allow overwriting labels layer {labels_layer_1}, with aligned result, please set overwrite to True, "
                "or specify a value for output_labels_layer."
            )

    translation = Translation([t1x, t1y], axes=("x", "y"))

    sdata = _add_label_layer(
        sdata,
        x_label_aligned,
        output_layer=output_labels_layer,
        chunks=chunks,
        transformation=translation,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    # only calculate shapes layer if it is specified
    if output_shapes_layer is not None:
        se_labels = _get_spatial_element(sdata, layer=output_labels_layer)

        # convert the labels to polygons and add them as shapes layer to sdata
        sdata = _add_shapes_layer(
            sdata,
            input=se_labels.data,
            output_layer=output_shapes_layer,
            transformation=translation,
            overwrite=overwrite,
        )

    return sdata


def _align_dask_arrays(
    x_label_1: Array,
    x_label_2: Array,
    **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
):
    # we will align labels of x_label_1 with labels of x_labels_2.

    assert (
        x_label_1.shape == x_label_2.shape
    ), "Only arrays with same shape are currently supported."

    chunks = kwargs.pop("chunks", None)
    depth = kwargs.pop("depth", {0: 100, 1: 100})
    boundary = kwargs.pop("boundary", "reflect")

    if chunks is None:
        assert (
            x_label_1.chunksize == x_label_2.chunksize
        ), "If chunks is not specified, please ensure Dask arrays have the same chunksize."

    _check_boundary(boundary)

    # make depth uniform + rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
    x_label_1, depth = _rechunk_overlap(
        x_label_1, depth=depth, chunks=chunks, spatial_dims=x_label_1.ndim
    )
    x_label_2, depth = _rechunk_overlap(
        x_label_2, depth=depth, chunks=chunks, spatial_dims=x_label_2.ndim
    )

    # output_chunks can be derived from either x_label_1 or x_label_2
    output_chunks = _add_depth_to_chunks_size(x_label_1.chunks, depth)

    x_labels = da.map_overlap(
        lambda m, f: _relabel_nuclei_with_cells_per_chunk(m, f),
        x_label_1,
        x_label_2,
        dtype=_SEG_DTYPE,
        allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
        chunks=output_chunks,  # e.g. ((1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60) ),
        depth=depth,
        trim=False,
        boundary="reflect",
        # this reflect is useless for this use case, but clean_up_masks and _trim_masks only support
        # results from map_overlap generated with "reflect", "nearest" and "constant"
    )

    x_labels = da.map_blocks(
        _clean_up_masks,
        x_labels,
        dtype=_SEG_DTYPE,
        depth=depth,
    )

    x_labels = _trim_masks(masks=x_labels, depth=depth)

    return x_labels


def _relabel_nuclei_with_cells_per_chunk(
    masks_nuclear: NDArray, masks_whole_cell: NDArray
) -> NDArray:
    nuclear_labels = np.unique(masks_nuclear)
    cell_labels = np.unique(masks_whole_cell)

    # Create a mapping array that holds the new labels for the nuclear mask.
    # The index of the array represents the original label, and the value at that index is the new label.
    mapping = np.zeros((nuclear_labels.max() + 1,), dtype=int)

    # only non-zero labels, no background
    nuclear_labels = nuclear_labels[nuclear_labels != 0]
    cell_labels = cell_labels[cell_labels != 0]

    # Now, we perform an operation to identify which cell each nucleus belongs to.
    # This creates a 2D array where each row corresponds to a nuclear label, and each column corresponds to a cell label.
    # The value is the count of the overlap area between that nucleus and cell.
    overlap_matrix = np.zeros(
        (nuclear_labels.max() + 1, cell_labels.max() + 1), dtype=int
    )
    np.add.at(overlap_matrix, (masks_nuclear.ravel(), masks_whole_cell.ravel()), 1)

    # For each nucleus, identify the cell with which it has the maximum overlap.
    # If a nucleus does not overlap with any cell, its label will be set to 0.
    for nuc_label in nuclear_labels:
        cell_with_max_overlap = np.argmax(overlap_matrix[nuc_label, :])
        if overlap_matrix[nuc_label, cell_with_max_overlap] == 0:
            cell_with_max_overlap = 0  # No overlap, set to 0

        mapping[nuc_label] = cell_with_max_overlap

    # Apply the mapping to the nuclear mask.
    relabeled_nuclei = mapping[masks_nuclear]

    return relabeled_nuclei
