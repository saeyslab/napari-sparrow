import dask.array as da
from dask.array import Array


def _nonzero_nonnan_percentile(array: Array, q: int) -> Array:
    """Computes the percentile of a dask array excluding all zeros and nans."""
    array = array.flatten()
    non_zero_non_nan_mask = (array != 0) & (~da.isnan(array))

    array = da.compress(non_zero_non_nan_mask, array)

    return da.percentile(array, q=q)[0]
