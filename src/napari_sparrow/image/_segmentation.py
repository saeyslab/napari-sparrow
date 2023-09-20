import itertools
import uuid
from typing import Any, Dict, Optional, Tuple, Callable

import dask.array as da
import numpy as np
import spatialdata
import squidpy as sq
import torch
from cellpose import models
from dask.array import Array
from dask.array.overlap import coerce_depth, ensure_minimum_chunksize
from numpy.typing import NDArray
from shapely.affinity import translate
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation

from napari_sparrow.image._image import _get_translation, _substract_translation_crd
from napari_sparrow.shape._shape import _mask_image_to_polygons

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


_SEG_DTYPE = np.uint32


def _cellpose(
    img,
    min_size=80,
    cellprob_threshold=-4,
    flow_threshold=0.85,
    diameter=100,
    model_type="cyto",
    channels=[1, 0],
    device="cpu",
):
    gpu = torch.cuda.is_available()
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
    output_layer="segmentation_mask",
    depth: Optional[Dict[int, int]]=None,
    chunks: Optional[str | int | tuple[int, ...]] = None,
    boundary: str = "reflect",
    **kwargs: Any, # keyword arguments to be passed to model
):
    
    # take the last image as layer to do next step in pipeline
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    if depth is None:
        log.warning(
            f"'depth' is equal to None, "
            f"If the image layer '{img_layer}' contains more than one chunk, "
            "this will lead to undesired effects at the borders of the chunks "
        )

    if not callable( model ):
        raise TypeError(f"Expected `model` to be a callable, found `{type(model)}`.")
    
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("boundary", boundary)
    kwargs.setdefault("chunks", chunks )

    segmentation_model=SegmentationModel( model )

    sdata = segmentation_model._segment_img_layer( sdata,
                                                    img_layer=img_layer,
                                                    output_layer=output_layer,
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
        output_layer:str="segmentation_masks",
        # chunks:Optional[str | int | tuple[int, ...]] = None,
        # depth: Dict[int, int]={0: 200, 1: 200},
        # boundary: str = "reflect",
        **kwargs: Any,
    ):
        # take the last image as layer to do next step in pipeline
        if img_layer is None:
            img_layer = [*sdata.images][-1]

        if "depth" not in kwargs.keys():
            log.warning(
                f"'depth' not specified, falling back to setting depth to None. "
                f"If the image layer '{img_layer}' contains more than one chunk, "
                "this will lead to undesired effects at the borders of the chunks "
            )

        chunks = kwargs.pop("chunks", None)
        depth = kwargs.pop("depth", None)
        boundary = kwargs.pop("boundary", "reflect")

        # take dask array and put channel dimension last
        x = sdata[img_layer].data.transpose(1, 2, 0)

        x_labels = self._segment(
            x, depth=depth, chunks=chunks, boundary=boundary, **kwargs
        )

        spatial_label = spatialdata.models.Labels2DModel.parse(x_labels)

        # TODO
        # if a crop is taken, need to add the crop to the offset.
        tx, ty = _get_translation(sdata[img_layer])
        # need to substract depth from translation, because otherwise labels layer not aligend with image
        if depth is not None:
            tx = tx - depth[1]
            ty = ty - depth[0]

        translation = Translation([tx, ty], axes=("x", "y"))

        set_transformation(spatial_label, translation)

        name_masks_with_overlap = f"_masks_{uuid.uuid4()}"

        # write to intermediate zarr here, to avoid race conditions, TODO check if necessary
        sdata.add_labels(name=name_masks_with_overlap, labels=spatial_label)

        masks = sdata[name_masks_with_overlap].data

        # this should not be done is depth is None
        masks = _trim_masks(masks=masks, depth=depth)

        spatial_label = spatialdata.models.Labels2DModel.parse(masks)

        # TODO
        # if a crop is taken, need to add the crop to the offset.
        tx, ty = _get_translation(sdata[img_layer])

        translation = Translation([tx, ty], axes=("x", "y"))

        set_transformation(spatial_label, translation)

        # during adding of image it is written to zarr store
        sdata.add_labels(name=output_layer, labels=spatial_label)

        return sdata

    def _segment(
        self,
        x: Array,
        depth: Dict[int, int],
        chunks: Optional[str | int | tuple[int, ...]] = None,
        boundary: str = "reflect",
        **kwargs: Any,
    ):
        _check_boundary(boundary)

        if chunks is not None:
            x = x.rechunk(chunks)

        # rechunk if new chunks are needed to fit depth in every chunk,
        # this allows us to send allow_rechunk=False with map_overlap,
        # and have control of chunk sizes of input dask array and output dask array
        depth2 = coerce_depth(x.ndim, depth)

        depths = [max(d) if isinstance(d, tuple) else d for d in depth2.values()]
        new_chunks = tuple(
            ensure_minimum_chunksize(size, c) for size, c in zip(depths, x.chunks)
        )

        x = x.rechunk(new_chunks)  # this is a no-op if x.chunks == new_chunks

        output_chunks = _add_depth_to_chunks_size(x.chunks, depth)

        shift = int(np.prod(x.numblocks) - 1).bit_length()

        # TODO probably better to pass depth and boundary explicitely to map_overlap, 
        # than it is clear that these are the kwargs to callable method
        kwargs.setdefault("depth", depth)
        kwargs.setdefault("boundary", boundary)

        # TODO pass these kwargs to segment function
        # kwargs = {
        #    "depth": depth,
        #    "boundary": boundary,
        #    "min_size": 80,
        #    "cellprob_threshold": -4,
        #    "flow_threshold": 0.85,
        #    "diameter": 85,
        #    "model_type": "cyto",
        #    "channels": [2, 1],
        #   "device": "cpu",
        # }

        # kwargs.setdefault("depth", {0: 30, 1: 30})
        # kwargs.setdefault("boundary", "reflect")

        x_labels = da.map_overlap(
            self._segment_chunk,
            x,
            dtype=_SEG_DTYPE,
            num_blocks=x.numblocks,
            shift=shift,
            drop_axis=x.ndim
            - 1,  # drop the last axis, i.e. the c-axis (only for determining output size of array)
            trim=False,
            allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
            chunks=output_chunks,  # e.g. ((1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60) ),
            # still need to send translation also to segment_chunk, because we need to do the translation that is on image layer 'clahe'
            **kwargs,
        )

        x_labels = da.map_blocks(
            _clean_up_masks,
            x_labels,
            dtype=_SEG_DTYPE,
            depth=depth,
        )

        return x_labels

    def _segment_chunk(
        self,
        block: NDArray,
        block_id: Tuple[int, ...],
        num_blocks: Tuple[int, ...],
        shift: int,
        **kwargs: Any,
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

        labels = self._model(block, **kwargs).astype(_SEG_DTYPE)
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

    # TODO check if it is not depth[0]+1 or block[ block.shape[0]-depth[0] ] -1, +1, double check
    crossing_masks = []
    if block_id[0] != 0:
        crossing_masks.append(block[depth[0]])
    if block_id[1] != 0:
        crossing_masks.append(block[:, depth[1]])
    if block_id[0] != total_blocks[0] - 1:
        crossing_masks.append(block[block.shape[0] - depth[0]])
    if block_id[1] != total_blocks[1] - 1:
        crossing_masks.append(block[:, block.shape[1] - depth[1]])

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


def segmentation_cellpose_deprecated(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels=[0, 0],
    chunks="auto",
    lazy=False,
    output_layer: str = "segmentation_mask",
) -> SpatialData:
    """
    Segment images using the Cellpose algorithm and add segmentation results to the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    img_layer : Optional[str], default=None
        The image layer in `sdata` to be segmented. If not provided, the last image layer in `sdata` is used.
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be segmented. It defines the bounds (x_min, x_max, y_min, y_max).
    device : str, default="cpu"
        Device to run Cellpose on, either "cpu" or "cuda" for GPU.
    min_size : int, default=80
        Minimum size of detected objects.
    flow_threshold : float, default=0.6
        Cellpose flow threshold.
    diameter : int, default=55
        Approximate diameter of objects to be detected.
    cellprob_threshold : int, default=0
        Cellpose cell probability threshold.
    model_type : str, default="nuclei"
        Type of model to be used in Cellpose, options include "nuclei" or "cyto".
    channels : list, default=[0, 0]
        Channels to use in Cellpose.
        For single channel images, the default value ([0,0]) should not be adapted.
        For multi channel images, the first element of the list is the channel to segment (count from 1),
        and the second element is the optional nuclear channel.
        E.g. for an image with PolyT in second channel, and DAPI in first channel use [2,1] if you want to segment PolyT + nuclei on DAPI;
        [2,0] if you only want to use PolyT and [1,0] if you only want to use DAPI."
    chunks : str, default="auto"
        The size of the chunks used by cellpose.
    lazy : bool, default=False
        If True, compute segmentation lazily.
    output_layer : str, default="segmentation_mask"
        The name of the label layer in which segmentation results will be stored in `sdata`.

    Returns
    -------
    SpatialData
        Updated sdata` object containing the segmentation mask and boundaries obtained from Cellpose.
        Segmentation masks will be added as a labels layer to the SpatialData object with name output_layer,
        and masks boundaries as a shapes layer with name f'{output_layer}_boundaries.'

    Raises
    ------
    ValueError
        If the chosen output_layer name contains the word 'filtered'.

    Notes
    -----
    The function integrates Cellpose segmentation into the SpatialData framework. It manages the pre and post-processing
    of data, translation of coordinates, and addition of segmentation results back to the SpatialData object.
    """

    if "filtered" in output_layer:
        raise ValueError(
            "Please choose an output_layer name that does not have 'filtered' in its name, "
            " as these are reserved for filtered out masks and shapes."
        )

    if img_layer is None:
        img_layer = [*sdata.images][-1]

    ic = sq.im.ImageContainer(sdata[img_layer], layer=img_layer)

    # crd is specified on original uncropped pixel coordinates
    # need to substract possible translation, because we use crd to crop imagecontainer, which does not take
    # translation into account
    if crd:
        crd = _substract_translation_crd(sdata[img_layer], crd)
    if crd:
        x0 = crd[0]
        x_size = crd[1] - crd[0]
        y0 = crd[2]
        y_size = crd[3] - crd[2]
        ic = ic.crop_corner(y=y0, x=x0, size=(y_size, x_size))

    tx, ty = _get_translation(sdata[img_layer])

    sq.im.segment(
        img=ic,
        layer=img_layer,
        method=_cellpose,
        channel=None,
        chunks=chunks,
        lazy=lazy,
        min_size=min_size,
        layer_added=output_layer,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        diameter=diameter,
        model_type=model_type,
        channels=channels,
        device=device,
        depth={0: 200, 1: 200},
    )

    if crd:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))

    temp = ic.data.rename_dims({"channels": "c"})
    spatial_label = spatialdata.models.Labels2DModel.parse(
        temp[output_layer].squeeze().transpose("y", "x")
    )

    set_transformation(spatial_label, translation)

    # during adding of image it is written to zarr store
    sdata.add_labels(name=output_layer, labels=spatial_label)

    polygons = _mask_image_to_polygons(mask=sdata[output_layer].data)

    x_translation, y_translation = _get_translation(sdata[output_layer])
    polygons["geometry"] = polygons["geometry"].apply(
        lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
    )

    shapes_name = f"{output_layer}_boundaries"

    sdata.add_shapes(
        name=shapes_name,
        shapes=spatialdata.models.ShapesModel.parse(polygons),
    )

    return sdata
