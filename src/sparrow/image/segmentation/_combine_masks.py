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
    sdata = combine_labels_layers(
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

    sdata = combine_labels_layers(
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


def combine_labels_layers(
    sdata: SpatialData,
    labels_layers: List[str],
    func: Callable[..., NDArray],
    depth: Tuple[int, ...] | int = 100,
    chunks: Optional[str | int | Tuple[int, ...]] = "auto",
    output_labels_layer: Optional[str] = None,
    output_shapes_layer: Optional[str] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
    relabel_chunks: bool = True,
    **kwargs: Any,  # keyword arguments to be passed to func
):
    fn_kwargs = kwargs

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
