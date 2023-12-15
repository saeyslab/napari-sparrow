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
    _fix_dimensions,
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
)
from napari_sparrow.image.segmentation._utils import (
    _SEG_DTYPE,
    _add_depth_to_chunks_size,
    _check_boundary,
    _clean_up_masks,
    _merge_masks,
    _rechunk_overlap,
    _substract_depth_from_chunks_size,
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
    depth: Tuple[int, int] | int = 100,
    chunks: Optional[str | int | Tuple[int, int]] = "auto",
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
        Callable should take as input numpy arrays of dimension (z,y,x,c) and return labels of dimension (z,y,x,c), with
        c dimension==1. It can have an arbitrary number of other parameters.
    output_labels_layer : str, default="segmentation_mask"
        Name of the label layer in which segmentation results will be stored in `sdata`.
    output_shapes_layer : Optional[str], default="segmentation_mask_boundaries"
        Name of the shapes layer where boundaries obtained output_labels_layer will be stored. If set to None, shapes won't be stored.
    depth : Tuple[int, int] | int, default=100
        The depth in y and x dimension. The depth parameter is passed to map_overlap. If trim is set to False,
        it's recommended to set the depth to a value greater than twice the estimated diameter of the cells/nulcei.
    chunks : Optional[str | int | Tuple[int, int]], default="auto"
        Chunk sizes for processing. Can be a string, integer or tuple of integers. If chunks is a Tuple,
        they  contain the chunk size that will be used in y and x dimension. Chunking in 'z' or 'c' dimension is not supported.
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
        # so we have ( z, y, x, c ), also do some checks on depth and chunks

        if se.data.ndim == 4:
            assert se.dims == (
                "c",
                "z",
                "y",
                "x",
            ), "dimension should be in order: ('c', 'z' , 'y', 'x')."
            # transpose x, so channel dimension is last
            x = _fix_dimensions(se.data, dims=se.dims, target_dims=("z", "y", "x", "c"))

        elif se.data.ndim == 3:
            assert se.dims == (
                "c",
                "y",
                "x",
            ), "dimension should be in order: ('c', 'y', 'x')."
            # transpose x, so channel dimension is last
            x = _fix_dimensions(se.data, dims=se.dims, target_dims=("y", "x", "c"))
            # add trivial z dimension.
            x = x[None, ...]
        else:
            raise ValueError(
                "Only 3D and 4D arrays are supported, i.e. (c, (z), y, x)."
            )

        if "depth" in kwargs:
            depth = kwargs["depth"]
            if isinstance(depth, int):
                kwargs["depth"] = {0: 0, 1: depth, 2: depth, 3: 0}
            else:
                assert (
                    len(depth) == x.ndim - 2
                ), "Please (only) provide depth for ( 'y', 'x')."
                # set depth for every dimension
                depth2 = {0: 0, 1: depth[0], 2: depth[1], 3: 0}
                kwargs["depth"] = depth2

        if "chunks" in kwargs:
            chunks = kwargs["chunks"]
            if chunks is not None:
                if not isinstance(chunks, (int, str)):
                    assert (
                        len(chunks) == x.ndim - 2
                    ), "Please (only) provide chunks for ( 'y', 'x')."
                    chunks = (x.shape[0], chunks[0], chunks[1], x.shape[-1])
                    kwargs["chunks"] = chunks

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
            fn_kwargs=fn_kwargs,
            **kwargs,
        )

        # squeeze the z-dim if it is 1 (i.e. case where you did not do 3D segmentation),
        # otherwise 2D labels layer would be saved as 3D
        if x_labels.shape[0] == 1:
            x_labels = x_labels.squeeze(0)

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
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
    ):
        chunks = kwargs.pop("chunks", None)
        depth = kwargs.pop("depth", {0: 0, 1: 100, 2: 100, 3: 0})
        assert len(depth) == 4, "Please provide depth for (('z', 'y', 'x', 'c'))"
        assert depth[0] == 0, "Depth not equal to 0 for 'z' dimension is not supported"
        assert depth[3] == 0, "Depth not equal to 0 for 'c' dimension is not supported"
        boundary = kwargs.pop("boundary", "reflect")
        trim = kwargs.pop("trim", False)

        if not trim and depth[1] == 0 or depth[2] == 0:
            log.warning(
                "Depth equal to zero not supported with trim==False, setting trim to True."
            )
            trim = True

        _check_boundary(boundary)

        # make depth uniform (dict with depth for z,y and x)
        # + rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
        x = _rechunk_overlap(
            x,
            depth=depth,
            chunks=chunks,
        )

        # remove trivial depth==0 for c dimension
        depth.pop(3)

        output_chunks = _add_depth_to_chunks_size(x.chunks, depth)
        # only support output chunks (i.e. labels) with channel shape == 1
        output_chunks = output_chunks[:-1] + (1,)

        # shift added to results of every chunk (i.e. if shift is 4, then 0 0 0 0 will be added to every label).
        # These 0's are then filled in with block_id number. This way labels are unique accross the different chunks.
        # not that if x.numblocks.bit_length() would be close to 16 bit, and each chunks has labels up to 16 bits,
        # this could lead to collisions.
        # ignore channel dim (num_blocks[3]), because channel dim of resulting label is supposed to be 1.
        num_blocks = x.numblocks
        shift = int(
            np.prod(num_blocks[0] * num_blocks[1] * num_blocks[2]) - 1
        ).bit_length()

        x_labels = da.map_overlap(
            self._segment_chunk,
            x,
            dtype=_SEG_DTYPE,
            num_blocks=num_blocks,
            shift=shift,
            trim=trim,
            allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
            chunks=output_chunks,  # e.g. ((7,) ,(1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60), (1,) ),
            depth=depth,
            boundary=boundary,
            fn_kwargs=fn_kwargs,
            **kwargs,
        )

        # For now, only support processing of x_labels with 1 channel dim
        x_labels = x_labels.squeeze(-1)

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
            x_labels = x_labels.rechunk(x_labels.chunksize)

        else:
            x_labels = da.map_blocks(
                _clean_up_masks,
                x_labels,
                dtype=_SEG_DTYPE,
                depth=depth,
                **kwargs,
            )

            output_chunks = _substract_depth_from_chunks_size(
                x_labels.chunks, depth=depth
            )

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

            x_labels = x_labels.rechunk(x_labels.chunksize)

        return x_labels

    def _segment_chunk(
        self,
        block: NDArray,
        block_id: Tuple[int, ...],
        num_blocks: Tuple[int, ...],
        shift: int,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> NDArray:
        if len(num_blocks) == 4:
            if num_blocks[0] != 1:
                raise ValueError(
                    f"Expected the number of blocks in the Z-dimension to be `1`, found `{num_blocks[0]}`."
                )
            if num_blocks[-1] != 1:
                raise ValueError(
                    f"Expected the number of blocks in the c-dimension to be `1`, found `{num_blocks[-1]}`."
                )

            # note: ignore num_blocks[3]==1 and block_id[3]==0, because we assume c-dimension is 1
            block_num = (
                block_id[0] * (num_blocks[1] * num_blocks[2])
                + block_id[1] * (num_blocks[2])
                + block_id[2]
            )

        else:
            raise ValueError(
                f"Expected `4` dimensional chunks, found `{len(num_blocks)}`."
            )

        labels = self._model(block, **fn_kwargs).astype(_SEG_DTYPE)
        mask: NDArray = labels > 0
        labels[mask] = (labels[mask] << shift) | block_num

        return labels
