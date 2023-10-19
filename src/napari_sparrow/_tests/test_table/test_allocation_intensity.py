import dask.array as da
import numpy as np

from napari_sparrow.table._allocation_intensity import _calculate_intensity


def test_calculate_intensity():
    chunk_size = (2, 2)

    mask_dask_array = da.from_array(
        np.array([[3, 0, 0, 0], [0, 3, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]]),
        chunks=chunk_size,
    )

    float_dask_array = da.from_array(
        np.array(
            [
                [0.5, 1.5, 2.5, 3.5],
                [4.5, 5.5, 6.5, 7.5],
                [8.5, 9.5, 10.5, 11.5],
                [12.5, 13.5, 14.5, 15.5],
            ]
        ),
        chunks=chunk_size,
    )

    sum_of_chunks = _calculate_intensity(
        mask_dask_array=mask_dask_array, float_dask_array=float_dask_array, chunks=chunk_size
    )

    expected_result = np.array([[51.5], [70.5], [6.0]])

    assert np.array_equal(sum_of_chunks, expected_result)
