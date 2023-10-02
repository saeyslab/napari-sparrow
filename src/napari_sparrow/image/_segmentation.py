import itertools
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import dask.array as da
import numpy as np
import spatialdata
import torch
from cellpose import models
from dask.array import Array
from dask.array.overlap import coerce_depth, ensure_minimum_chunksize
from numpy.typing import NDArray
from shapely.affinity import translate
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from napari_sparrow.image._image import (
    _add_label_layer,
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
)
from napari_sparrow.shape._shape import _mask_image_to_polygons
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_SEG_DTYPE = np.uint32


def _cellpose(
    img: NDArray,
    min_size: int = 80,
    cellprob_threshold: int = 0,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
    device: str = "cpu",
) -> NDArray:
    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    model = models.Cellpose(gpu=gpu, model_type=model_type, device=torch.device(device))
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks


def segment(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    model: Callable[..., NDArray] = _cellpose,
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
    scale_factors
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

        # take dask array and put channel dimension last
        x = se.data.transpose(1, 2, 0)

        # crd is specified on original uncropped pixel coordinates
        # need to substract possible translation, because we use crd to crop dask array, which does not take
        # translation into account
        if crd:
            crd = _substract_translation_crd(se, crd)
            x = x[crd[2] : crd[3], crd[0] : crd[1], :]

        x_labels = self._segment(
            x,
            fn_kwargs,
            **kwargs,
        )

        tx, ty = _get_translation(se)

        if crd:
            tx = tx + crd[0]
            ty = ty + crd[2]

        translation = Translation([tx, ty], axes=("x", "y"))

        sdata=_add_label_layer(
            sdata,
            arr=x_labels,
            output_layer=output_labels_layer,
            chunks=x_labels.chunksize,
            transformation=translation,
            scale_factors=scale_factors,
            overwrite=overwrite,
        )

        # only calculate shapes layer if is specified
        if output_shapes_layer is not None:
            se_labels = _get_spatial_element(sdata, layer=output_labels_layer)
            # now calculate the polygons
            polygons = _mask_image_to_polygons(mask=se_labels.data)

            x_translation, y_translation = _get_translation(se_labels)
            polygons["geometry"] = polygons["geometry"].apply(
                lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
            )

            sdata.add_shapes(
                name=output_shapes_layer,
                shapes=spatialdata.models.ShapesModel.parse(polygons),
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
        depth = kwargs.pop("depth", {0: 100, 1: 100})
        boundary = kwargs.pop("boundary", "reflect")
        trim = kwargs.pop("trim", False)

        _check_boundary(boundary)

        if chunks is not None:
            x = x.rechunk(chunks)

        # rechunk if new chunks are needed to fit depth in every chunk,
        # this allows us to send allow_rechunk=False with map_overlap,
        # and have control of chunk sizes of input dask array and output dask array
        if isinstance(depth, list):
            depth = tuple(depth)
        if isinstance(depth, int):
            depth = (x.ndim - 1) * (depth,)
        depth2 = coerce_depth(x.ndim, depth)

        for i in range(2):
            if depth2[i] > x.chunksize[i]:
                log.warning(
                    f"Depth at index {i} exceeds chunk size. Adjusting to a quarter of chunk size: {x.chunksize[i]/4}"
                )
                depth2[i] = int(x.chunksize[i] // 4)

        depths = [max(d) if isinstance(d, tuple) else d for d in depth2.values()]
        new_chunks = tuple(
            ensure_minimum_chunksize(size + 1, c) for size, c in zip(depths, x.chunks)
        )

        # we don't want channel dimension in depth (coerce_depth added this dimension).
        last_key = list(depth2.keys())[-1]
        depth2.pop(last_key)
        depth = depth2

        x = x.rechunk(new_chunks)  # this is a no-op if x.chunks == new_chunks

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
            chunks=output_chunks,  # e.g. ((1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60) ),
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


def _clean_up_masks(
    block: NDArray,
    block_id: tuple[int, ...],
    block_info,
    depth,
) -> NDArray:
    total_blocks = block_info[0]["num-chunks"]

    # get the 'inside' region of the block, i.e. the original chunk without depth appended
    y_start, y_stop = depth[0], block.shape[0] - depth[0]
    x_start, x_stop = depth[1], block.shape[1] - depth[1]

    # get indices of all adjacent blocks
    adjacent_blocks = _get_ajdacent_block_ids(block_id, total_blocks)

    # get all masks id's that cross the boundary of original chunk (without depth appended)
    # masks that are on the boundary of the larger array (e.g. y==depth[0] axis are skipped)

    crossing_masks = []
    if block_id[0] != 0:
        crossing_masks.append(block[depth[0]])
    if block_id[1] != 0:
        crossing_masks.append(block[:, depth[1]])
    if block_id[0] != total_blocks[0] - 1:
        crossing_masks.append(block[block.shape[0] - depth[0]])
    if block_id[1] != total_blocks[1] - 1:
        crossing_masks.append(block[:, block.shape[1] - depth[1]])

    if crossing_masks:
        crossing_masks = np.unique(np.hstack(crossing_masks))

    def calculate_area(crd, mask_position):
        return np.sum(
            (crd[0] <= mask_position[0])
            & (mask_position[0] < crd[1])
            & (crd[2] <= mask_position[1])
            & (mask_position[1] < crd[3])
        )

    for mask_label in crossing_masks:
        if mask_label == 0:
            continue
        mask_position = np.where(block == mask_label)

        inside_region = calculate_area(
            (y_start, y_stop, x_start, x_stop), mask_position
        )

        for adjacent_block_id in adjacent_blocks:
            crd = _calculate_boundary_adjacent_block(
                block, depth, block_id, adjacent_block_id
            )

            outside_region = calculate_area(crd, mask_position)

            # if intersection with mask and region outside chunk is bigger than inside region, set values of chunk to 0 for this masks.
            # For edge case where inside region and outside region is the same, it will be assigned to both chunks.
            # Because we write out final masks single threaded, this is no issue.
            # Note that is better that both chunks claim the masks, than that no chunks are claiming the mask. If they both claim the mask,
            # It will be assigned to the 'last' chunk, while writing to zarr store.
            if outside_region > inside_region:
                block[block == mask_label] = 0
                break

    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    subset = block[
        depth[0] : block.shape[0] - depth[0], depth[1] : block.shape[1] - depth[1]
    ]
    # Unique masks gives you all masks that are in 'original' array (i.e. without depth added)
    unique_masks = np.unique(subset)

    # Create a mask for labels that are NOT in the subset
    mask = ~np.isin(block, unique_masks)
    block[mask] = 0

    return block


def _trim_masks(masks: Array, depth: Dict[int, int]) -> Array:
    # now create final array
    chunk_coords = list(
        itertools.product(
            *[range(0, s, cs) for s, cs in zip(masks.shape, masks.chunksize)]
        )
    )
    chunk_ids = [
        (y // masks.chunksize[0], x // masks.chunksize[1]) for (y, x) in chunk_coords
    ]

    chunks = _substract_depth_from_chunks_size(masks.chunks, depth=depth)

    masks_trimmed = da.zeros((sum(chunks[0]), sum(chunks[1])), chunks=chunks, dtype=int)

    for chunk_id, chunk_coord in zip(chunk_ids, chunk_coords):
        chunk = masks.blocks[chunk_id]

        mask_chunk_shape = chunk.shape

        y_start = chunk_coord[0]
        x_start = chunk_coord[1]

        # trim labels if chunk lays on boundary of larger array
        if y_start == 0:
            chunk = chunk[depth[0] :, :]
        if x_start == 0:
            chunk = chunk[:, depth[1] :]
        if (y_start + mask_chunk_shape[0]) == masks.shape[0]:
            chunk = chunk[: -depth[0], :]
        if (x_start + mask_chunk_shape[1]) == masks.shape[1]:
            chunk = chunk[:, : -depth[1]]

        # now convert back to non-overlapping coordinates.

        # check if this is correct TODO, thinks so
        y_offset = max(0, y_start - (chunk_id[0] * 2 * depth[0] + depth[0]))
        x_offset = max(0, x_start - (chunk_id[1] * 2 * depth[1] + depth[1]))

        non_zero_mask = chunk != 0

        # Update only the non-zero positions in the dask array
        masks_trimmed[
            y_offset : y_offset + chunk.shape[0], x_offset : x_offset + chunk.shape[1]
        ] = da.where(
            non_zero_mask,
            chunk,
            masks_trimmed[
                y_offset : y_offset + chunk.shape[0],
                x_offset : x_offset + chunk.shape[1],
            ],
        )

    return masks_trimmed


def _check_boundary(boundary: str) -> None:
    valid_boundaries = ["reflect", "periodic", "nearest"]

    if boundary not in valid_boundaries:
        raise ValueError(
            f"'{boundary}' is not a valid boundary. It must be one of {valid_boundaries}."
        )


def _add_depth_to_chunks_size(
    chunks: Tuple[Tuple[int, ...], ...], depth: Dict[int, int]
):
    result = []
    for i, item in enumerate(chunks):
        if i in depth:  # check if there's a corresponding depth value
            result.append(tuple(x + depth[i] * 2 for x in item))
    return tuple(result)


def _substract_depth_from_chunks_size(
    chunks: Tuple[Tuple[int, ...], ...], depth: Dict[int, int]
):
    result = []
    for i, item in enumerate(chunks):
        if i in depth:  # check if there's a corresponding depth value
            result.append(tuple(x - depth[i] * 2 for x in item))
    return tuple(result)


def _get_ajdacent_block_ids(block_id, total_blocks):
    y, x = block_id
    max_y, max_x = total_blocks

    potential_neighbors = [
        (y - 1, x - 1),  # top-left
        (y, x - 1),  # top
        (y + 1, x - 1),  # top-right
        (y - 1, x),  # left
        (y + 1, x),  # right
        (y - 1, x + 1),  # bottom-left
        (y, x + 1),  # bottom
        (y + 1, x + 1),  # bottom-right
    ]

    # Filter out neighbors that have negative IDs or exceed the total number of blocks
    neighbors = [
        neighbor
        for neighbor in potential_neighbors
        if 0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y
    ]
    return neighbors


def _calculate_boundary_adjacent_block(chunk, depth, block_id, adjacent_block_id):
    if adjacent_block_id[0] > block_id[0]:
        y_start = chunk.shape[0] - depth[0]
        y_stop = chunk.shape[0]
    elif adjacent_block_id[0] == block_id[0]:
        y_start = depth[0]
        y_stop = chunk.shape[0] - depth[0]
    else:
        y_start = 0
        y_stop = depth[0]

    if adjacent_block_id[1] > block_id[1]:
        x_start = chunk.shape[1] - depth[1]
        x_stop = chunk.shape[1]
    elif adjacent_block_id[1] == block_id[1]:
        x_start = depth[1]
        x_stop = chunk.shape[1] - depth[1]
    else:
        x_start = 0
        x_stop = depth[1]

    return (y_start, y_stop, x_start, x_stop)
