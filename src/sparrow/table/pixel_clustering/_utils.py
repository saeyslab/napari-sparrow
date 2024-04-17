import dask.array as da
import numpy as np
from dask.array import Array
from numpy.typing import NDArray


def _nonzero_nonnan_percentile(array: Array, q: int) -> Array:
    """Computes the percentile of a dask array excluding all zeros."""
    array = array.flatten()
    non_zero_non_nan_mask = (array != 0) & (~da.isnan(array))

    array = da.compress(non_zero_non_nan_mask, array)

    return da.percentile(array, q=q)[0]


def _get_non_nan_pixel_values_and_location(array: Array) -> NDArray:
    """
    Get non nan pixel values

    Function gets all non nan values from a dask array, and saves them to a numpy array
    resulting numpy array has following columns:
    pixel_value_channel_0, pixel_value_channel_1, pixel_value_channel_2,...pixel_location_z, pixel_location_y, pixel_location_x
    ...
    Array can be used as .X of AnnData object, which can then e.g. be used as input to flowsom
    """
    # c, z, y, x
    assert array.ndim == 4

    # Create a boolean mask where True corresponds to non-NaN values
    non_nan_mask = ~da.isnan(array)

    # Get the indices where the mask is True (non-NaN values)
    non_nan_indices = da.nonzero(non_nan_mask)

    # Use this mask to filter out the non-NaN values
    non_nan_values = array[non_nan_mask]

    computed_indices_c, computed_indices_z, computed_indices_y, computed_indices_x, computed_pixel_value = da.compute(
        non_nan_indices[0],  # channel indices
        non_nan_indices[1],  # z indices
        non_nan_indices[2],  # y indices
        non_nan_indices[3],  # x indices
        non_nan_values,
    )

    c_indices_list = np.unique(computed_indices_c)
    results = []
    for c in c_indices_list:
        results.append(computed_pixel_value[computed_indices_c == c])

    # now add z, y and x coordinate, should be same location for all channels
    results.append(computed_indices_z[computed_indices_c == c_indices_list[0]])
    results.append(computed_indices_y[computed_indices_c == c_indices_list[0]])
    results.append(computed_indices_x[computed_indices_c == c_indices_list[0]])
    return np.column_stack(results)  # c, z, y, x
