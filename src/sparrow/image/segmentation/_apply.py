from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple

import dask.array as da
import numpy as np
from dask.array import Array
from numpy.typing import NDArray
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


def apply_labels_layers(
    sdata: SpatialData,
    func: Callable[..., NDArray | Array],
    labels_layers: List[str] | str,
    output_labels_layer: Optional[str] = None,
    output_shapes_layer: Optional[str] = None,
    depth: Tuple[int, ...] | int = 100,
    chunks: str | int | Tuple[int, ...] = "auto",
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
    relabel_chunks: bool = True,
    **kwargs: Any,  # keyword arguments to be passed to func
):
    """
    Apply a specified function to a labels layer in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the labels layer to be processed.
    func : Callable[..., NDArray | Array]
        The Callable to apply to the labels layer.
    labels_layer : List[str] | str.
        The labels layer(s) in `sdata` to process.
    output_labels_layer : Optional[str], default=None.
        The name of the output labels layer where results will be stored. This must be specified.
    output_shapes_layer : Optional[str], default=None.
        The name of the output shapes layer where results will be stored.
    depth : Tuple[int, ...], default=100.
        The depth around the boundary of each block to load when the array is split into blocks
        (for alignment). This ensures that the split isn't causing misalignment along the edges.
        Default is 100. Please set depth>cell size to avoid chunking effects.
    chunks : str | Tuple[int, ...] | int, default=None.
        Specification for rechunking the data before applying the function.
        If chunks is a Tuple, they should contain desired chunk size for 'y', 'x'.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite : bool, default=False
        If True, overwrites the output layer if it already exists in `sdata`.
    relabel_chunks: bool, default=True.
        Whether to relabel the labels of each chunk after being processed by func. If set to True, a bit shift will be applied, ensuring no collisions.
    **kwargs : Any
        Keyword arguments to be passed to func.

    Returns
    -------
    SpatialData
        The `sdata` object with the processed labels layer added to the specified output layer.
        If `output_shapes_layer` is provided, a shapes layer will be created corresponding to this labels layer.

    Raises
    ------
    ValueError
        - If `output_labels_layer` is not provided.
        - If `chunks` is a Tuple, and does not match (y,x).
        - If `depth` is a Tuple, and does not match (y,x).
        - If a label layer in `labels_layer` can not be found.
        - If number of blocks in z-dimension is not equal to 1.

    Notes
    -----
    This function is designed for processing labels layers stored in a SpatialData object using dask for potential
    parallelism and out-of-core computation. It takes care of relabeling across chunks, to avoid collisions.
    """

    fn_kwargs = kwargs

    labels_layers = (
        list(labels_layers)
        if isinstance(labels_layers, Iterable) and not isinstance(labels_layers, str)
        else [labels_layers]
    )

    if output_labels_layer is None:
        raise ValueError("Please specify a name for the output layer.")

    # first do the precondition.
    def _get_layers(
        sdata: SpatialData, labels_layers: List[str]
    ) -> Tuple[List[Array], Translation]:
        """
        Process multiple labels layers and return the label data (list of dask arrays)
        and the translation associated with the dask arrays.
        """
        # sanity check
        for layer in labels_layers:
            if layer not in [*sdata.labels]:
                raise ValueError(
                    f"Layer '{layer}' not found in available label layers '{[*sdata.labels]}' of sdata."
                )
        labels_data = []

        # Initial checks for the first layer to set a reference for comparison
        first_se = _get_spatial_element(sdata, layer=labels_layers[0])
        first_x_label = first_se.data
        first_translation = _get_translation(first_se)

        for layer in labels_layers:
            se = _get_spatial_element(sdata, layer=layer)
            x_label = se.data
            translation = _get_translation(se)

            # Ensure the shape is the same as the first label layer
            assert x_label.shape == first_x_label.shape, (
                f"Only arrays with same shape are currently supported, "
                f"but labels layer with name {layer} has shape {x_label.shape}, "
                f"while the first labels layer has shape {first_x_label.shape}"
            )

            # Ensure the translation is the same as the first label layer
            assert translation == first_translation, (
                f"Labels layer with name {layer} should "
                f"have the same translation as the first labels layer."
            )

            labels_data.append(x_label)

        translation = Translation(
            [first_translation[0], first_translation[1]], axes=("x", "y")
        )

        return labels_data, translation

    labels_arrays, translation = _get_layers(sdata, labels_layers=labels_layers)

    # kwargs to be passed to map_overlap/map_blocks
    kwargs = {}
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("chunks", chunks)

    # labels_arrays is a list of dask arrays
    # do some processing on the labels
    array = _combine_dask_arrays(
        labels_arrays,
        relabel_chunks=relabel_chunks,
        func=func,
        fn_kwargs=fn_kwargs,
        **kwargs,
    )

    translation

    sdata = _add_label_layer(
        sdata,
        array,
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


def _combine_dask_arrays(
    labels_arrays: Iterable[Array],
    relabel_chunks: bool,
    func: Callable[..., NDArray],
    fn_kwargs: Mapping[str, Any] = MappingProxyType(
        {}
    ),  # keyword arguments to be passed to func
    **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
) -> Array:
    # combines a list of dask arrays

    if not labels_arrays:
        raise ValueError("No arrays provided")

    # Check for consistent shapes
    first_shape = labels_arrays[0].shape
    for x_label in labels_arrays:
        assert (
            x_label.shape == first_shape
        ), "Only arrays with same shape are currently supported."

    chunks = kwargs.pop("chunks", None)
    depth = kwargs.pop("depth", 100)
    boundary = kwargs.pop("boundary", "reflect")

    # First make dimension uniform (z,y,x).
    _to_squeeze = False
    _labels_arrays = []
    for i, x_label in enumerate(labels_arrays):
        if x_label.ndim == 2:
            _to_squeeze = True
            _labels_arrays.append(x_label[None, ...])
        else:
            _labels_arrays.append(x_label)

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

    rechunked_arrays = []
    for i, x_label in enumerate(_labels_arrays):
        #  rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
        x_label = _rechunk_overlap(x_label, depth=depth, chunks=chunks)
        assert (
            x_label.numblocks[0] == 1
        ), f"Expected the number of blocks in the Z-dimension to be `1`, found `{x_label.numblocks[0]}`."

        if i == 0:
            # output_chunks can be derived from any rechunked x_label in labels_arrays
            output_chunks = _add_depth_to_chunks_size(x_label.chunks, depth)

        rechunked_arrays.append(x_label)

    # num_blocks is same for all arrays
    num_blocks = rechunked_arrays[0].numblocks
    shift = int(np.prod(num_blocks[0] * num_blocks[1] * num_blocks[2]) - 1).bit_length()

    x_labels = da.map_overlap(
        lambda *arrays, block_id=None, **kw: _process_masks(
            *arrays, block_id=block_id, **kw
        ),  # Unpack and pass all arrays to _process_masks
        *rechunked_arrays,  # Unpack the list of Dask arrays as individual arguments
        dtype=_SEG_DTYPE,
        trim=False,  # we do not trim, but we clean up and merge in subsequent steps.
        allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
        chunks=output_chunks,  # e.g. ((7,) ,(1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60), (1,) ),
        depth=depth,
        boundary=boundary,
        num_blocks=num_blocks,
        shift=shift,
        relabel_chunks=relabel_chunks,
        _func=func,  # _func will be passed to _process_masks
        fn_kwargs=fn_kwargs,  # keyword arguments to be passed to func
        **kwargs,  # additional kwargs passed to map_overlap
    )

    x_labels = da.map_blocks(
        _clean_up_masks,
        x_labels,
        dtype=_SEG_DTYPE,
        depth=depth,
        **kwargs,
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
        boundary=boundary,
        _depth=depth,
    )

    x_labels = x_labels.rechunk(x_labels.chunksize)

    # squeeze if a trivial dimension was added.
    if _to_squeeze:
        x_labels = x_labels.squeeze(0)

    return x_labels


def _process_masks(
    *arrays: NDArray,
    block_id: Tuple[int, ...],
    num_blocks: Tuple[int, ...],
    shift: int,
    relabel_chunks: bool,
    _func: Callable,
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
):
    if len(num_blocks) == 3:
        if num_blocks[0] != 1:
            raise ValueError(
                f"Expected the number of blocks in the Z-dimension to be `1`, found `{num_blocks[0]}`."
            )
        block_num = (
            block_id[0] * (num_blocks[1] * num_blocks[2])
            + block_id[1] * (num_blocks[2])
            + block_id[2]
        )

    else:
        raise ValueError(f"Expected `3` dimensional chunks, found `{len(num_blocks)}`.")

    x_label = _func(*arrays, **fn_kwargs)

    if relabel_chunks:
        mask: NDArray = x_label > 0
        x_label[mask] = (x_label[mask] << shift) | block_num

    else:
        log.warning(
            f"Chunks are not relabeled. "
            f"Please make sure that provided Callable {_func} returns unique labels across chunks, otherwise collisions can be expected."
        )

    return x_label