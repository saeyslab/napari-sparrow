import numpy as np

from sparrow.table.pixel_clustering._create_pixel_matrix import _sampling_function


def test_sampling_function():
    array = np.array([[[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], [[10, 11, np.nan], [12, np.nan, 13], [14, 15, 16]]])

    array = array[:, None, ...]

    result_array = _sampling_function(block=array, seed=10, fraction=0.5)

    correct_array = np.array(
        [
            [[[np.nan, 2.0, np.nan], [np.nan, np.nan, 6.0], [np.nan, 8.0, np.nan]]],
            [[[np.nan, 11.0, np.nan], [np.nan, np.nan, 13.0], [np.nan, 15.0, np.nan]]],
        ]
    )

    assert np.allclose(result_array, correct_array, equal_nan=True)
