from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

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
    _substract_translation_crd,
)
from napari_sparrow.image.segmentation._utils import (
    _SEG_DTYPE,
    _add_depth_to_chunks_size,
    _check_boundary,
    _clean_up_masks,
    _rechunk_overlap,
    _trim_masks,
)
from napari_sparrow.image.segmentation.segmentation_models._cellpose import (
    _cellpose as _model,
)
from napari_sparrow.shape._shape import _add_shapes_layer
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def segment(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    model: Callable[..., NDArray] = _model,
    output_labels_layer: str = "segmentation_mask",
    output_shapes_layer: Optional[str] = "segmentation_mask_boundaries",
    depth: Tuple[int, int] = (100, 100),
    chunks: Optional[str | int | tuple[int, ...]] = "auto",
    boundary: str = "reflect",
    trim: bool = False,
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
    **kwargs: Any,
):
    """
    Segment images using a provided model and add segmentation results
    (labels layer and shapes layer) to the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the image layer to segment.
    img_layer : Optional[str], default=None
        The image layer in `sdata` to be segmented. If not provided, the last image layer in `sdata` is used.
    model : Callable[..., NDArray], default=_cellpose
        The segmentation model function used to process the images.
        Should take as input arrays of dimension (y,x,c) and return labels of dimension (y,x)
    output_labels_layer : str, default="segmentation_mask"
        Name of the label layer in which segmentation results will be stored in `sdata`.
    output_shapes_layer : Optional[str], default="segmentation_mask_boundaries"
        Name of the shapes layer where boundaries obtained output_labels_layer will be stored. If set to None, shapes won't be stored.
    depth : Tuple[int, int], default=(100, 100)
        The depth parameter to be passed to map_overlap. If trim is set to False,
        it's recommended to set the depth to a value greater than twice the estimated diameter of the cells/nulcei.
    chunks : Optional[str | int | tuple[int, ...]], default="auto"
        Chunk sizes for processing. Can be a string, integer or tuple of integers.
    boundary : str, default="reflect"
        Boundary parameter passed to map_overlap.
    trim : bool, default=False
        If set to True, overlapping regions will be processed using the `squidpy` algorithm.
        If set to False, the `sparrow` algorithm will be employed instead. For dense cell distributions,
        we recommend setting trim to True.
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be segmented. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors : Optional[ScaleFactors_t], optional
        Scale factors to apply for multiscale.
    overwrite : bool, default=False
        If True, overwrites the existing layers if they exist. Otherwise, raises an error if the layers exist.
    **kwargs : Any
        Additional keyword arguments passed to the provided `model`.

    Returns
    -------
    SpatialData
        Updated `sdata` object containing the segmentation results.

    Raises
    ------
    TypeError
        If the provided `model` is not callable.
    """

    fn_kwargs = kwargs

    # take the last image as layer to do next step in pipeline
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    if not callable(model):
        raise TypeError(f"Expected `model` to be a callable, found `{type(model)}`.")

    # kwargs to be passed to map_overlap/map_blocks
    kwargs = {}
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("boundary", boundary)
    kwargs.setdefault("chunks", chunks)
    kwargs.setdefault("trim", trim)

    segmentation_model = SegmentationModel(model)

    sdata = segmentation_model._segment_img_layer(
        sdata,
        img_layer=img_layer,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        crd=crd,
        scale_factors=scale_factors,
        overwrite=overwrite,
        fn_kwargs=fn_kwargs,
        **kwargs,
    )
    return sdata


class SegmentationModel:
    def __init__(
        self,
        model: Callable[..., NDArray],
    ):
        self._model = model

    def _segment_img_layer(
        self,
        sdata: SpatialData,
        img_layer: Optional[str] = None,
        output_labels_layer: str = "segmentation_mask",
        output_shapes_layer: Optional[str] = "segmentation_mask_boundaries",
        crd: Optional[Tuple[int, int, int, int]] = None,
        scale_factors: Optional[ScaleFactors_t] = None,
        overwrite: bool = False,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ):
        if img_layer is None:
            img_layer = [*sdata.images][-1]

        se = _get_spatial_element(sdata, layer=img_layer)

        # take dask array and put channel dimension last,
        # so we have ( z, y, x, c ).
        if se.data.ndim == 4:
            x = se.data.transpose(1, 2, 3, 0)
        elif se.data.ndim == 3:
            x = se.data.transpose(1, 2, 0)
            # add trivial dimension z dimension.
            x = x[None, ...]
        else:
            raise ValueError(
                "Only 3D and 4D arrays are supported, i.e. (c, (z), y, x)."
            )

        # TODO support crd for 3D
        # crd is specified on original uncropped pixel coordinates
        # need to substract possible translation, because we use crd to crop dask array, which does not take
        # translation into account
        if crd is not None:
            crd = _substract_translation_crd(se, crd)
            if crd is not None:
                x = x[:, crd[2] : crd[3], crd[0] : crd[1], :]
                x = x.rechunk(x.chunksize)

        x_labels = self._segment(
            x,
            dims=( "z", "y", "x", "c" ),
            fn_kwargs=fn_kwargs,
            **kwargs,
        )

        # squeeze the z-dim if it is 1 (i.e. case where you did not do 3D segmentation)
        if x_labels.shape[0] == 1:
            x_labels=x_labels.squeeze(0)

        tx, ty = _get_translation(se)

        if crd is not None:
            tx = tx + crd[0]
            ty = ty + crd[2]

        translation = Translation([tx, ty], axes=("x", "y"))

        sdata = _add_label_layer(
            sdata,
            arr=x_labels,
            output_layer=output_labels_layer,
            chunks=x_labels.chunksize,
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

    def _segment(
        self,
        x: Array,
        dims: Tuple[str,...]=( "z", "y", "x", "c" ),
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
    ):
        chunks = kwargs.pop("chunks", None)
        depth = kwargs.pop("depth", {0: 100, 1: 100})
        boundary = kwargs.pop("boundary", "reflect")
        trim = kwargs.pop("trim", False)

        _check_boundary(boundary)

        # make depth uniform + rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
        x, depth = _rechunk_overlap(
            x,
            depth=depth,
            chunks=chunks,
            dims=dims,
        )

        output_chunks = _add_depth_to_chunks_size(x.chunks, depth)

        shift = int(np.prod(x.numblocks) - 1).bit_length()

        x_labels = da.map_overlap(
            self._segment_chunk,
            x,
            dtype=_SEG_DTYPE,
            num_blocks=x.numblocks,
            shift=shift,
            drop_axis=x.ndim
            - 1,  # drop the last axis, i.e. the c-axis (only for determining output size of array)
            trim=trim,
            allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
            chunks=output_chunks,  # e.g. ((7,) ,(1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60) ),
            depth=depth,
            boundary=boundary,
            fn_kwargs=fn_kwargs,
            **kwargs,
        )

        # if trim==True --> use squidpy's way of handling neighbouring blocks
        if trim:
            from dask_image.ndmeasure._utils._label import (
                connected_components_delayed,
                label_adjacency_graph,
                relabel_blocks,
            )

            # max because labels are not continuous (and won't be continuous)
            label_groups = label_adjacency_graph(x_labels, None, x_labels.max())
            new_labeling = connected_components_delayed(label_groups)
            x_labels = relabel_blocks(x_labels, new_labeling)

        else:
            x_labels = da.map_blocks(
                _clean_up_masks,
                x_labels,
                dtype=_SEG_DTYPE,
                depth=depth,
                **kwargs,
            )

            #return x_labels
            #x_labels.compute()
            # test without compute here...
            x_labels = _trim_masks(masks=x_labels, depth=depth)

        return x_labels

    def _segment_chunk(
        self,
        block: NDArray,
        block_id: Tuple[int, ...],
        num_blocks: Tuple[int, ...],
        shift: int,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> NDArray:
        if len(num_blocks) == 2:
            block_num = block_id[0] * num_blocks[1] + block_id[1]
        elif len(num_blocks) == 3:
            block_num = (
                block_id[0] * (num_blocks[1] * num_blocks[2])
                + block_id[1] * num_blocks[2]
            )
        elif len(num_blocks) == 4:
            if num_blocks[-1] != 1:
                raise ValueError(
                    f"Expected the number of blocks in the Z-dimension to be `1`, found `{num_blocks[-1]}`."
                )
            block_num = (
                block_id[0] * (num_blocks[1] * num_blocks[2])
                + block_id[1] * num_blocks[2]
            )
        else:
            raise ValueError(
                f"Expected either `2`, `3` or `4` dimensional chunks, found `{len(num_blocks)}`."
            )

        labels = self._model(block, **fn_kwargs).astype(_SEG_DTYPE)
        mask: NDArray = labels > 0
        labels[mask] = (labels[mask] << shift) | block_num

        return labels
