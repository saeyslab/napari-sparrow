from __future__ import annotations

from typing import Any, Optional, Tuple

import dask.array as da
import numpy as np
from dask.array import Array
from numpy.typing import NDArray
from skimage.segmentation import expand_labels
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from sparrow.image._image import (
    _add_label_layer,
    _get_spatial_element,
    _get_translation,
)
from sparrow.image.segmentation._utils import (
    _SEG_DTYPE,
    _add_depth_to_chunks_size,
    _check_boundary,
    _clean_up_masks,
    _merge_masks,
    _rechunk_overlap,
    _substract_depth_from_chunks_size,
)
from sparrow.shape._shape import _add_shapes_layer
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def expand_labels_layer(
    sdata: SpatialData,
    labels_layer: str,
    distance: int = 10,
    depth: Tuple[int, int] | int = 100,
    chunks: Optional[str | int | Tuple[int, int]] = "auto",
    output_labels_layer: Optional[str] = None,
    output_shapes_layer: Optional[str] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
):
    """
    Expaned cells in labels layer of Spatialdata object with `distance`, using `skimage.segmentation.expand_labels`.

    Parameters
    ----------
    sdata : SpatialData
        The spatial data object containing the labels layer to be expanded.
    labels_layer : str
        The name of the labels layer to be expanded.
    distance: int
        distance passed to skimage.segmentation.expand_labels.
    depth : Tuple[int, int], optional
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell size + distance to avoid chunking effects.
    chunks : Optional[str | int | Tuple[int, int]], optional
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
    ValueError
        If 'output_labels_layer' is None and 'overwrite' is False, indicating ambiguity in the
        user's intent.

    Notes
    -----
    The function works with Dask arrays and can handle large datasets that don't fit into memory.


    Examples
    --------
    >>> sdata = expand_labels_layer(
            sdata,
            labels_layer='layer',
            distance=10,
            depth=(100, 100),
            chunks=(1024, 1024),
            output_labels_layer='layer_expanded',
            output_shapes_layer='layer_expanded_boundaries',
            overwrite=True,
        )
    """

    se = _get_spatial_element(sdata, layer=labels_layer)

    x_label = se.data

    t1x, t1y = _get_translation(se)

    x_label_expanded = _expand_dask_array(
        x_label, chunks=chunks, depth=depth, distance=distance
    )

    if output_labels_layer is None:
        output_labels_layer = labels_layer
        if overwrite == False:
            raise ValueError(
                "output_labels_layer was set to None, but overwrite to False. "
                f"to allow overwriting labels layer {labels_layer}, with aligned result, please set overwrite to True, "
                "or specify a value for output_labels_layer."
            )

    translation = Translation([t1x, t1y], axes=("x", "y"))

    sdata = _add_label_layer(
        sdata,
        x_label_expanded,
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


def _expand_dask_array(
    x_label: Array,
    **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
):
    # we will expand x_label
    distance = kwargs.pop("distance", 10)
    chunks = kwargs.pop("chunks", None)
    depth = kwargs.pop("depth", 100 + distance)
    boundary = kwargs.pop("boundary", "reflect")

    _to_squeeze = False
    if x_label.ndim == 2:
        _to_squeeze = True
        x_label = x_label[None, ...]

    if isinstance(depth, int):
        depth = {0: 0, 1: depth, 2: depth}
    else:
        assert (
            len(depth) == x_label.ndim - 1
        ), "Please (only) provide depth for ( 'y', 'x')."
        # set depth for every dimension
        depth2 = {0: 0, 1: depth[0], 2: depth[1]}
        depth = depth2

    if chunks is not None:
        if not isinstance(chunks, (int, str)):
            assert (
                len(chunks) == x_label.ndim - 1
            ), "Please (only) provide chunks for ( 'y', 'x')."
            chunks = (x_label.shape[0], chunks[0], chunks[1])

    _check_boundary(boundary)

    #  rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
    x_label = _rechunk_overlap(x_label, depth=depth, chunks=chunks)

    # TDDO x_label.numblocks[0] should be x_label.shape[0]
    assert (
        x_label.numblocks[0] == 1
    ), f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label.numblocks[0]}`."

    output_chunks = _add_depth_to_chunks_size(x_label.chunks, depth)

    x_labels = da.map_overlap(
        _expand_cells,
        x_label,
        dtype=_SEG_DTYPE,
        allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
        chunks=output_chunks,  # e.g. ((1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60) ),
        depth=depth,
        trim=False,
        boundary="reflect",
        distance=distance,
        **kwargs,
        # this reflect is useless for this use case, but clean_up_masks and _merge_masks only support
        # results from map_overlap generated with "reflect", "nearest" and "constant"
    )

    x_labels = da.map_blocks(
        _clean_up_masks,
        x_labels,
        dtype=_SEG_DTYPE,
        depth=depth,
    )

    output_chunks = _substract_depth_from_chunks_size(x_labels.chunks, depth=depth)

    x_labels = da.map_overlap(
        _merge_masks,
        x_labels,
        dtype=_SEG_DTYPE,
        num_blocks=x_labels.numblocks,
        trim=False,
        allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
        chunks=output_chunks,  # e.g. ((7,) ,(1024, 1024, 452), (1024, 1024, 452), (1,) ),
        depth=depth,
        boundary="reflect",
        _depth=depth,
    )

    # squeeze if a trivial dimension was added.
    if _to_squeeze:
        x_labels = x_labels.squeeze(0)

    return x_labels


def _expand_cells(
    x_label: NDArray,
    distance: int,
) -> NDArray:
    # input and output is numpy array of shape (z,y,x)

    assert x_label.ndim == 3
    x_label = expand_labels(x_label, distance=distance)

    return x_label
