from __future__ import annotations

import numpy as np
from dask.array import Array
from dask.array.overlap import ensure_minimum_chunksize
from numpy.typing import NDArray

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_SEG_DTYPE = np.uint32


def _rechunk_overlap(
    x: Array,
    depth: dict[int, int],
    chunks: str | int | tuple[int, ...] | None = "auto",
) -> Array:
    # rechunk, so that we ensure minimum overlap

    assert (
        len(depth) == x.ndim
    ), f"Please provide depth value for every dimension of x ({x.ndim}). Provided depth was '{depth}'"

    if chunks is not None:
        x = x.rechunk(chunks)

    # rechunk if new chunks are needed to fit depth in every chunk,
    # this allows us to send allow_rechunk=False with map_overlap,
    # and have control of chunk sizes of input dask array and output dask array

    for i in range(len(depth)):
        if depth[i] != 0:
            if depth[i] > x.chunksize[i]:
                log.warning(
                    f"Depth for dimension {i} exceeds chunk size. Adjusting to a quarter of chunk size: {x.chunksize[i]/4}"
                )
                depth[i] = int(x.chunksize[i] // 4)

    new_chunks = tuple(ensure_minimum_chunksize(size + 1, c) for size, c in zip(depth.values(), x.chunks))

    x = x.rechunk(new_chunks)  # this is a no-op if x.chunks == new_chunks

    return x


def _clean_up_masks_exact(
    block: NDArray,
    _depth_1: dict[int, int],
    _depth_2: dict[int, int],
    block_id: tuple[int, int, int],
    block_info,
) -> NDArray:
    total_blocks = block_info[0]["num-chunks"]
    assert (
        total_blocks[0] == 1
    ), "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
    assert _depth_1[0] == 0, "Depth not equal to 0 in z dimension is currently not supported."
    assert len(_depth_1) == 3, "Please provide depth values for z,y and x."
    assert (
        block_id[0] == 0
    ), "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."

    crossing_masks = set()
    crossing_masks_horizontal_low = set()  # low in the block, i.e. low y indices
    crossing_masks_horizontal_high = set()  # high in the block, i.e. high y indices
    crossing_masks_vertical_left = set()  # left in the block, i.e. low x indices
    crossing_masks_vertical_right = set()  # right in the block, i.e. high x indices

    """
    We assume following position of chunks (z,y,x).
    (0, 0, 0)  (0, 0, 1)
    (0, 1, 0)  (0, 1, 1)
    """
    # remove part that was added to the block due to the 'reflect' in segmentation. In this way calculated center will always be center.
    if block_id[1] == 0:
        block[:, : _depth_1[1] + _depth_2[1], :] = 0
    if block_id[1] + 1 == total_blocks[1]:
        block[:, -(_depth_1[1] + _depth_2[1]) :, :] = 0
    if block_id[2] == 0:
        block[:, :, : _depth_1[2] + _depth_2[2]] = 0
    if block_id[2] + 1 == total_blocks[2]:
        block[:, :, -(_depth_1[2] + _depth_2[2]) :] = 0

    if block_id[1] > 0:
        crossing_masks_horizontal_low.update(
            np.unique(block[:, _depth_1[1] + _depth_2[1], _depth_1[2] + _depth_2[2] : -(_depth_1[2] + _depth_2[2])])
        )
    if block_id[1] + 1 < total_blocks[1]:
        crossing_masks_horizontal_high.update(
            np.unique(block[:, -(_depth_1[1] + _depth_2[1]), _depth_1[2] + _depth_2[2] : -(_depth_1[2] + _depth_2[2])])
        )
    if block_id[2] > 0:
        crossing_masks_vertical_left.update(
            np.unique(block[:, _depth_1[1] + _depth_2[1] : -(_depth_1[1] + _depth_2[1]), _depth_1[2] + _depth_2[2]])
        )
    if block_id[2] + 1 < total_blocks[2]:
        crossing_masks_vertical_right.update(
            np.unique(block[:, _depth_1[1] + _depth_2[1] : -(_depth_1[1] + _depth_2[1]), -(_depth_1[2] + _depth_2[2])])
        )

    crossing_masks = (
        crossing_masks_horizontal_low
        | crossing_masks_horizontal_high
        | crossing_masks_vertical_left
        | crossing_masks_vertical_right
    )

    mask_to_reset = []  # these are masks that should be set to 0
    for mask_label in crossing_masks:
        if mask_label == 0:
            continue

        # put a condition on the mask position, i.e. they should lie within the orignal image + depth_1, otherwise they also contain
        # the mask position of neighbouring chunk (in _depth_2) if relabel_chunks==False. If relabel_chunks==True, we could ignore this relabeling.
        y_min, y_max = _depth_2[1], block.shape[1] - (_depth_2[1])
        x_min, x_max = _depth_2[2], block.shape[2] - (_depth_2[2])
        mask_position = np.where(block == mask_label)
        mask_position = _subset_mask_position(
            mask_position=mask_position, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )

        total_area = len(mask_position[0])

        if not total_area:
            mask_to_reset.append(mask_label)
            continue

        center = (np.mean(mask_position[0]), np.mean(mask_position[1]), np.mean(mask_position[2]))
        # check if center is in the block
        claims_original = _center_in_block(block=block, center=center, _depth_1=_depth_1, _depth_2=_depth_2)

        # get mask position of adjacent chunks
        claims_list = []
        total_area_adjacent_list = []

        # we could put this back into a function
        if mask_label in crossing_masks_horizontal_high:
            translation_direction = (0, 1, 0)
            result = _check_claims_adjacent_block(
                block=block,
                mask_position=mask_position,
                _depth_1=_depth_1,
                _depth_2=_depth_2,
                translation_direction=translation_direction,
            )
            if result is not None:
                claims, total_area_adjacent = result
                claims_list.append(claims)
                total_area_adjacent_list.append(total_area_adjacent)
            if mask_label in crossing_masks_vertical_left:
                translation_direction = (0, 1, -1)
                result = _check_claims_adjacent_block(
                    block=block,
                    mask_position=mask_position,
                    _depth_1=_depth_1,
                    _depth_2=_depth_2,
                    translation_direction=translation_direction,
                )
                if result is not None:
                    claims, total_area_adjacent = result
                    claims_list.append(claims)
                    total_area_adjacent_list.append(total_area_adjacent)
            elif mask_label in crossing_masks_vertical_right:
                translation_direction = (0, 1, 1)
                result = _check_claims_adjacent_block(
                    block=block,
                    mask_position=mask_position,
                    _depth_1=_depth_1,
                    _depth_2=_depth_2,
                    translation_direction=translation_direction,
                )
                if result is not None:
                    claims, total_area_adjacent = result
                    claims_list.append(claims)
                    total_area_adjacent_list.append(total_area_adjacent)
        elif mask_label in crossing_masks_horizontal_low:
            translation_direction = (0, -1, 0)
            result = _check_claims_adjacent_block(
                block=block,
                mask_position=mask_position,
                _depth_1=_depth_1,
                _depth_2=_depth_2,
                translation_direction=translation_direction,
            )
            if result is not None:
                claims, total_area_adjacent = result
                claims_list.append(claims)
                total_area_adjacent_list.append(total_area_adjacent)
            if mask_label in crossing_masks_vertical_left:
                translation_direction = (0, -1, -1)
                result = _check_claims_adjacent_block(
                    block=block,
                    mask_position=mask_position,
                    _depth_1=_depth_1,
                    _depth_2=_depth_2,
                    translation_direction=translation_direction,
                )
                if result is not None:
                    claims, total_area_adjacent = result
                    claims_list.append(claims)
                    total_area_adjacent_list.append(total_area_adjacent)
            elif mask_label in crossing_masks_vertical_right:
                translation_direction = (0, -1, 1)
                result = _check_claims_adjacent_block(
                    block=block,
                    mask_position=mask_position,
                    _depth_1=_depth_1,
                    _depth_2=_depth_2,
                    translation_direction=translation_direction,
                )
                if result is not None:
                    claims, total_area_adjacent = result
                    claims_list.append(claims)
                    total_area_adjacent_list.append(total_area_adjacent)
        if mask_label in crossing_masks_vertical_left:
            translation_direction = (0, 0, -1)
            result = _check_claims_adjacent_block(
                block=block,
                mask_position=mask_position,
                _depth_1=_depth_1,
                _depth_2=_depth_2,
                translation_direction=translation_direction,
            )
            if result is not None:
                claims, total_area_adjacent = result
                claims_list.append(claims)
                total_area_adjacent_list.append(total_area_adjacent)
        elif mask_label in crossing_masks_vertical_right:
            translation_direction = (0, 0, 1)
            result = _check_claims_adjacent_block(
                block=block,
                mask_position=mask_position,
                _depth_1=_depth_1,
                _depth_2=_depth_2,
                translation_direction=translation_direction,
            )
            if result is not None:
                claims, total_area_adjacent = result
                claims_list.append(claims)
                total_area_adjacent_list.append(total_area_adjacent)

        # for empty list claims_list, np.sum( claims_list ) will return False
        # now decide which masks to set to 0.
        claims_list = np.array(claims_list)
        total_area_adjacent_list = np.array(total_area_adjacent_list)
        # now decide on which masks to reset
        if claims_original:
            # case where adjacent chunk also claim mask, we should only retain it if it is the largest.
            if np.sum(claims_list):
                total_area_adjacent_max = np.max(total_area_adjacent_list[claims_list])
                if total_area < total_area_adjacent_max:
                    mask_to_reset.append(mask_label)
        else:
            if np.sum(claims_list):
                mask_to_reset.append(mask_label)
            # others also do not claim it, only keep if it is the largest
            else:
                if len(total_area_adjacent_list) > 0:
                    total_area_adjacent_max = np.max(total_area_adjacent_list)
                    if total_area < total_area_adjacent_max:
                        mask_to_reset.append(mask_label)
                else:
                    pass
                    # log.warning(
                    #    f"For cell with mask label {mask_label} in chunk with block id {block_id} there where no matches found in other chunks. "
                    #    "This indicates disagreement between chunks. Consider increasing depth or chunk size."
                    # )
                    # case where there is no matching cell found in any of the adjacent chunks,
                    # then we do not reset, although this typically indicates disagreement between chunks, so for algorithm to give good results, this better not happen offen.
                    # maybe print warning here, also for debugging

    mask = np.isin(block, mask_to_reset)
    block[mask] = 0

    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    subset = block[
        :,
        (_depth_1[1] + _depth_2[1]) : block.shape[1] - (_depth_1[1] + _depth_2[1]),
        (_depth_1[2] + _depth_2[2]) : block.shape[2] - (_depth_1[2] + _depth_2[2]),
    ]
    # Unique masks gives you all masks that are (at least partially) in 'original' array (i.e. without depth added)
    unique_masks = np.unique(subset)

    # Create a mask for labels that are NOT in the subset
    mask = ~np.isin(block, unique_masks)
    block[mask] = 0

    # trim _depth_2 from block
    return block[:, _depth_2[1] : -_depth_2[1], _depth_2[2] : -_depth_2[2]]


def _subset_mask_position(
    mask_position: tuple[NDArray, NDArray, NDArray], y_min: int, y_max: int, x_min: int, x_max: int
) -> tuple[NDArray, NDArray, NDArray]:
    condition = (
        (mask_position[1] >= y_min)
        & (mask_position[1] < y_max)
        & (mask_position[2] >= x_min)
        & (mask_position[2] < x_max)
    )

    return mask_position[0][condition], mask_position[1][condition], mask_position[2][condition]


def _check_claims_adjacent_block(
    block: NDArray,
    mask_position: tuple[NDArray, NDArray, NDArray],
    _depth_1: dict[int, int],
    _depth_2: dict[int, int],
    translation_direction: tuple[int, int, int],
) -> tuple[bool, int]:
    # translation=(0, _depth_1[1] + int(_depth_2[1]/2) , 0 )
    translation = tuple(
        a * b
        for a, b in zip(
            translation_direction,
            (
                _depth_1[0] + int(_depth_2[0] / 2),
                _depth_1[1] + int(_depth_2[1] / 2),
                _depth_1[2] + int(_depth_2[2] / 2),
            ),
        )
    )

    # these are the coordinates of the adjacent block (in original block coordinate system), but with _depth_1
    coordinates_adjacent_block_with_depth = _get_coordinates(
        block=block,
        _depth_1=_depth_1,
        _depth_2={0: 0, 1: int(_depth_2[1] / 2), 2: int(_depth_2[2] / 2)},
        translation_direction=translation_direction,
    )

    # get the center of the conflicting cell in other chunk, plus the total area of that cell
    result = _get_center_and_area(
        block=block,
        mask_position=mask_position,
        translation=translation,
        coordinates_adjacent_block_with_depth=coordinates_adjacent_block_with_depth,  # these coordinates are used for subsetting mask_positions
    )
    # If result is None, this means that in adjacent chunk (via translation), no match is found.
    # This is actually a sign of conflicting segmentation results, maybe we should log a warning in that case.
    if result is None:
        return None
    # coordinates_claims_by_adjacent_block are not the coordinates of the adjacent block, but a region outside current block. If center_adjacent_chunk lies in it,
    # the mask is claimed by the adjacent block.
    coordinates_claims_by_adjacent_block = _get_coordinates(
        block=block,
        _depth_1=_depth_1,
        _depth_2=_depth_2,
        translation_direction=translation_direction,
    )

    center_adjacent_block, total_area_adjacent = result

    y_min, y_max, x_min, x_max = coordinates_claims_by_adjacent_block
    if (
        center_adjacent_block[1] >= y_min
        and center_adjacent_block[1] < y_max
        and center_adjacent_block[2] >= x_min
        and center_adjacent_block[2] < x_max
    ):
        claims = True
    else:
        claims = False

    return claims, total_area_adjacent


def _center_in_block(block: NDArray, center: tuple[float, float, float], _depth_1, _depth_2) -> bool:
    y_min, y_max = _depth_1[1] + _depth_2[1], block.shape[1] - (_depth_1[1] + _depth_2[1])
    x_min, x_max = _depth_1[2] + _depth_2[2], block.shape[2] - (_depth_1[2] + _depth_2[2])

    if center[1] >= y_min and center[1] < y_max and center[2] >= x_min and center[2] < x_max:
        return True

    return False


def _get_center_and_area(
    block: NDArray,
    mask_position: tuple[NDArray, NDArray, NDArray],
    translation: tuple[int, int, int],
    coordinates_adjacent_block_with_depth: tuple[
        int, int, int, int
    ],  # these coordinates are necessary for subsetting mask_positions.
) -> tuple[tuple[float, float, float], int] | None:
    """Helper function for _clean_up_masks"""
    total_area = len(mask_position[0])
    y_min, y_max, x_min, x_max = coordinates_adjacent_block_with_depth
    # get mask position of adjacent chunk
    mask_position_adjacent_chunk = [
        mask_position[0] + translation[0],
        mask_position[1] + translation[1],
        mask_position[2] + translation[2],
    ]
    mask_labels_adjacent_chunk = np.unique(
        block[mask_position_adjacent_chunk[0], mask_position_adjacent_chunk[1], mask_position_adjacent_chunk[2]]
    )
    total_area_adjacent_list = []
    center_adjacent_list = []
    intersection_area_list = []
    for mask_label_adjacent in mask_labels_adjacent_chunk:
        if mask_label_adjacent == 0:
            continue
        # get the label with maximum overlap
        _mask_position_adjacent_chunk = np.where(block == mask_label_adjacent)
        _mask_position_adjacent_chunk = _subset_mask_position(
            _mask_position_adjacent_chunk,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
        )
        total_area_adjacent = len(_mask_position_adjacent_chunk[0])
        if not total_area_adjacent:
            continue
        # we translate back, because we need to calculate overlap with adjacent.
        _mask_position_adjacent_chunk = [
            _mask_position_adjacent_chunk[0] - translation[0],
            _mask_position_adjacent_chunk[1] - translation[1],
            _mask_position_adjacent_chunk[2] - translation[2],
        ]
        center_adjacent = (
            np.mean(_mask_position_adjacent_chunk[0]),
            np.mean(_mask_position_adjacent_chunk[1]),
            np.mean(_mask_position_adjacent_chunk[2]),
        )
        # need to check overlap. Check if overlap more than half with other
        intersection_area = _size_interection_mask_position(_mask_position_adjacent_chunk, mask_position)
        # should be at least half in each other to be considered segmentation result from same cell, this way there will be relative good agreement across all chunks
        # about who claims the mask
        # TODO to match conflicting cells, here some improvement is still possible.
        if intersection_area / total_area < 0.5 or intersection_area / total_area_adjacent < 0.5:
            continue
        intersection_area_list.append(intersection_area)
        total_area_adjacent_list.append(total_area_adjacent)
        center_adjacent_list.append(center_adjacent)
    # do a return if no intersection is found
    if not intersection_area_list:
        return
    max_intersection_id = np.argmax(intersection_area_list)
    return center_adjacent_list[max_intersection_id], total_area_adjacent_list[max_intersection_id]


def _get_coordinates(
    block: NDArray, _depth_1: dict[int, int], _depth_2: dict[int, int], translation_direction: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    # get coordinates of adjacent block
    if translation_direction[1] == 1:
        y_min = (
            block.shape[1] - (_depth_1[1] + _depth_2[1])
        )  # maybe we should do _depth_2[1]/2, and then not do translation, no not possible, because we need to do calc of overlap
        y_max = block.shape[1]
    elif translation_direction[1] == -1:
        y_min = 0
        y_max = _depth_1[1] + _depth_2[1]
    elif translation_direction[1] == 0:
        # then just take original chunk
        y_min = _depth_1[1] + _depth_2[1]
        y_max = block.shape[1] - (_depth_1[1] + _depth_2[1])
    else:
        raise ValueError("Translation direction should be one of the following values: 1, 0, -1.")

    if translation_direction[2] == 1:
        x_min = block.shape[2] - (_depth_1[2] + _depth_2[2])
        x_max = block.shape[2]
    elif translation_direction[2] == -1:
        x_min = 0
        x_max = _depth_1[2] + _depth_2[2]
    elif translation_direction[2] == 0:
        # if translation direction is 0, then just take original chunk
        x_min = _depth_1[2] + _depth_2[2]
        x_max = block.shape[2] - (_depth_1[2] + _depth_2[2])
    else:
        raise ValueError("Translation direction should be one of the following values: 1, 0, -1.")

    return y_min, y_max, x_min, x_max


def _size_interection_mask_position(
    mask_position1: tuple[NDArray, NDArray, NDArray], mask_position2: tuple[NDArray, NDArray, NDArray]
) -> int:
    # Convert each mask_position to a set of tuples (coordinate pairs)
    set1 = set(zip(mask_position1[0], mask_position1[1], mask_position1[2]))
    set2 = set(zip(mask_position2[0], mask_position2[1], mask_position2[2]))
    # Find the intersection of these sets
    intersection = set1.intersection(set2)
    # The size of the overlap is the size of the intersection
    overlap_size = len(intersection)

    return overlap_size


def _clean_up_masks(
    block: NDArray,
    block_id: tuple[int, int, int],
    block_info,
    depth: dict[int, int],
) -> NDArray:
    total_blocks = block_info[0]["num-chunks"]
    assert (
        total_blocks[0] == 1
    ), "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
    total_blocks = total_blocks[1:]
    assert depth[0] == 0, "Depth not equal to 0 in z dimension is currently not supported."
    assert len(depth) == 3, "Please provide depth values for z,y and x."

    # remove z-dimension from depth
    depth[0] = depth[1]
    depth[1] = depth[2]
    del depth[2]

    # get the 'inside' region of the block, i.e. the original chunk without depth appended
    y_start, y_stop = depth[0], block.shape[1] - depth[0]
    x_start, x_stop = depth[1], block.shape[2] - depth[1]  # TODO fix bug, should be depth[1]

    assert (
        block_id[0] == 0
    ), "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
    block_id = block_id[1:]

    # get indices of all adjacent blocks
    adjacent_blocks = _get_ajdacent_block_ids(block_id, total_blocks)

    # get all masks id's that cross the boundary of original chunk (without depth appended)
    # masks that are on the boundary of the larger array (e.g. y==depth[0] axis are skipped)

    crossing_masks = set()

    if block_id[0] != 0:
        crossing_masks.update(np.unique(block[:, depth[0], :]))
    if block_id[1] != 0:
        crossing_masks.update(np.unique(block[:, :, depth[1]]))
    if block_id[0] != total_blocks[0] - 1:
        crossing_masks.update(np.unique(block[:, block.shape[1] - depth[0], :]))
    if block_id[1] != total_blocks[1] - 1:
        crossing_masks.update(np.unique(block[:, :, block.shape[2] - depth[1]]))

    def calculate_area(crd, mask_position):
        return np.sum(
            (crd[0] <= mask_position[0])
            & (mask_position[0] < crd[1])
            & (crd[2] <= mask_position[1])
            & (mask_position[1] < crd[3])
        )

    masks_to_reset = []
    for mask_label in crossing_masks:
        if mask_label == 0:
            continue
        mask_position = np.where(block == mask_label)
        # not interested in which z-slice these mask_positions are
        mask_position = mask_position[1:]

        inside_region = calculate_area((y_start, y_stop, x_start, x_stop), mask_position)

        for adjacent_block_id in adjacent_blocks:
            crd = _calculate_boundary_adjacent_block(block.shape[1:], depth, block_id, adjacent_block_id)

            outside_region = calculate_area(crd, mask_position)

            # if intersection with mask and region outside chunk is bigger than inside region, set values of chunk to 0 for this masks.
            # For edge case where inside region and outside region is the same, it will be assigned to both chunks.
            # Note that is better that both chunks claim the masks, than that no chunks are claiming the mask.
            if outside_region > inside_region:
                masks_to_reset.append(mask_label)
                break

    mask = np.isin(block, masks_to_reset)
    block[mask] = 0

    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    subset = block[:, depth[0] : block.shape[1] - depth[0], depth[1] : block.shape[2] - depth[1]]
    # Unique masks gives you all masks that are in 'original' array (i.e. without depth added)
    unique_masks = np.unique(subset)

    # Create a mask for labels that are NOT in the subset
    mask = ~np.isin(block, unique_masks)
    block[mask] = 0

    return block


def _merge_masks(
    array: NDArray,
    _depth: dict[int, int],
    num_blocks: tuple[int, int, int],
    block_id: tuple[int, int, int],
) -> NDArray:
    # helper function to merge the chunks

    assert (
        num_blocks[0] == 1
    ), "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."

    assert _depth[0] == 0, "Depth not equal to 0 in z dimension is currently not supported."
    assert len(_depth) == 3, "Please provide depth values for z,y and x."

    new_array = array[:, _depth[1] * 2 : -_depth[1] * 2, _depth[2] * 2 : -_depth[2] * 2]
    # y,x
    # upper ( y, x+1 )
    if block_id[2] + 1 != num_blocks[2]:
        overlap = array[:, _depth[1] * 2 : -_depth[1] * 2, -_depth[2] :]
        sliced_new_array = new_array[
            :,
            :,
            -_depth[2] :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            :,
            -_depth[2] :,
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # upper right ( y+1, x+1 )
    if block_id[1] + 1 != num_blocks[1] and block_id[2] + 1 != num_blocks[2]:
        overlap = array[:, -_depth[1] :, -_depth[2] :]
        sliced_new_array = new_array[
            :,
            -_depth[1] :,
            -_depth[2] :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            -_depth[1] :,
            -_depth[2] :,
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # right ( y+1, x )
    if block_id[1] + 1 != num_blocks[1]:
        overlap = array[:, -_depth[1] :, _depth[2] * 2 : -_depth[2] * 2]
        sliced_new_array = new_array[:, -_depth[1] :, :]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[:, -_depth[1] :, :] = np.where(non_zero_mask, overlap, sliced_new_array)
    # under right ( y+1, x-1 )
    if block_id[1] + 1 != num_blocks[1] and block_id[2] != 0:
        overlap = array[:, -_depth[1] :, : _depth[2]]
        sliced_new_array = new_array[
            :,
            -_depth[1] :,
            : _depth[2],
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            -_depth[1] :,
            : _depth[2],
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # lower ( y, x-1 )
    if block_id[2] != 0:
        overlap = array[:, _depth[1] * 2 : -_depth[1] * 2, : _depth[2]]
        sliced_new_array = new_array[
            :,
            :,
            : _depth[2],
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            :,
            : _depth[2],
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # lower left ( y-1, x-1 )
    if block_id[1] != 0 and block_id[2] != 0:
        overlap = array[:, : _depth[1], : _depth[2]]
        sliced_new_array = new_array[
            :,
            : _depth[1],
            : _depth[2],
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            : _depth[1],
            : _depth[2],
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    #  left ( y-1, x )
    if block_id[1] != 0:
        overlap = array[:, : _depth[1], _depth[2] * 2 : -_depth[2] * 2]
        sliced_new_array = new_array[
            :,
            : _depth[1],
            :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            : _depth[1],
            :,
        ] = np.where(non_zero_mask, overlap, sliced_new_array)
    # upper left ( y-1, x+1 )
    if block_id[1] != 0 and block_id[2] + 1 != num_blocks[2]:
        overlap = array[:, : _depth[1], -_depth[2] :]
        sliced_new_array = new_array[
            :,
            : _depth[1],
            -_depth[2] :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            : _depth[1],
            -_depth[2] :,
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )

    return new_array


def _check_boundary(boundary: str) -> None:
    valid_boundaries = ["reflect", "periodic", "nearest"]

    if boundary not in valid_boundaries:
        raise ValueError(f"'{boundary}' is not a valid boundary. It must be one of {valid_boundaries}.")


def _add_depth_to_chunks_size(chunks: tuple[tuple[int, ...], ...], depth: dict[int, int, int]):
    result = []
    for i, item in enumerate(chunks):
        if i in depth:  # check if there's a corresponding depth value
            result.append(tuple(x + depth[i] * 2 for x in item))
        else:
            result.append(item)
    return tuple(result)


def _substract_depth_from_chunks_size(chunks: tuple[tuple[int, ...], ...], depth: dict[int, int]):
    result = []
    for i, item in enumerate(chunks):
        if i in depth:  # check if there's a corresponding depth value
            result.append(tuple(x - depth[i] * 2 for x in item))
        else:
            result.append(item)
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
    neighbors = [neighbor for neighbor in potential_neighbors if 0 <= neighbor[0] < max_y and 0 <= neighbor[1] < max_x]
    return neighbors


def _calculate_boundary_adjacent_block(chunk_shape, depth, block_id, adjacent_block_id):
    if adjacent_block_id[0] > block_id[0]:
        y_start = chunk_shape[0] - depth[0]
        y_stop = chunk_shape[0]
    elif adjacent_block_id[0] == block_id[0]:
        y_start = depth[0]
        y_stop = chunk_shape[0] - depth[0]
    else:
        y_start = 0
        y_stop = depth[0]

    if adjacent_block_id[1] > block_id[1]:
        x_start = chunk_shape[1] - depth[1]
        x_stop = chunk_shape[1]
    elif adjacent_block_id[1] == block_id[1]:
        x_start = depth[1]
        x_stop = chunk_shape[1] - depth[1]
    else:
        x_start = 0
        x_stop = depth[1]

    return (y_start, y_stop, x_start, x_stop)


def _get_block_position(
    chunks: tuple[tuple[int, ...], ...], block_id: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    """
    Given a block structure of a 4D Dask array and a block ID, return the start and stop positions in the full array for that block for the 1st (y) and 2nd (x) dimension.

    Parameters
    ----------
    block_structure: A tuple of tuples, where each inner tuple
                            represents the sizes of the blocks in that dimension.
    block_id: A tuple representing the position of the block in the
                     block structure.

    Returns
    -------
    A tuple (y_start, y_stop, x_start, x_stop)
    """
    y_structure, x_structure = chunks[1], chunks[2]
    _, i, j, _ = block_id

    y_start = sum(y_structure[:i])
    y_stop = y_start + y_structure[i]

    x_start = sum(x_structure[:j])
    x_stop = x_start + x_structure[j]

    return y_start, y_stop, x_start, x_stop
