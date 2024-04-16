import dask.array as da
import numpy as np

from sparrow.table.pixel_clustering._utils import _get_non_nan_pixel_values_and_location


def test_get_non_nan_pixel_values_and_location():
    array = da.from_array(
        [[[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], [[10, 11, np.nan], [12, np.nan, 13], [14, 15, 16]]],
    )
    # add trivial z-dimension
    array = array[:, None, ...]

    non_nan_array = _get_non_nan_pixel_values_and_location(array)

    # correct_array
    # pixel_value_c_0, pixel_value_c_1, z_pixel_location, y_pixel_location, x_pixel_location
    correct_array = np.array(
        [
            [1.0, 10.0, 0.0, 0.0, 0.0],
            [2.0, 11.0, 0.0, 0.0, 1.0],
            [4.0, 12.0, 0.0, 1.0, 0.0],
            [6.0, 13.0, 0.0, 1.0, 2.0],
            [7.0, 14.0, 0.0, 2.0, 0.0],
            [8.0, 15.0, 0.0, 2.0, 1.0],
            [9.0, 16.0, 0.0, 2.0, 2.0],
        ]
    )

    assert np.array_equal(non_nan_array, correct_array)
