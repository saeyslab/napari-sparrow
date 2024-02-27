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
    x_start, x_stop = depth[0], block.shape[2] - depth[1]

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
