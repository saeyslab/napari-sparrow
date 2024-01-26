from __future__ import annotations

from typing import Optional, Tuple

from numpy.typing import NDArray
from skimage.segmentation import expand_labels
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image.segmentation._merge_masks import apply_labels_layers
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

    sdata = apply_labels_layers(
        sdata,
        labels_layers=[labels_layer],
        func=_expand_cells,
        depth=depth,
        chunks=chunks,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        scale_factors=scale_factors,
        overwrite=overwrite,
        relabel_chunks=False,
        distance=distance,
    )

    return sdata


def _expand_cells(
    x_label: NDArray,
    distance: int,
) -> NDArray:
    # input and output is numpy array of shape (z,y,x)

    assert x_label.ndim == 3
    x_label = expand_labels(x_label, distance=distance)

    return x_label
