import numpy as np

from sparrow.table.pixel_clustering._create_pixel_matrix import _sampling_function, create_pixel_matrix


def test_create_pixel_matrix(sdata_multi_c):
    sdata_multi_c = create_pixel_matrix(
        sdata_multi_c,
        img_layer=["raw_image"],
        output_table_layer="table_pixels",
        channels=[2, 5, 7, 20],
        q=99,
        q_sum=5,
        q_post=99.9,
        sigma=2.0,
        norm_sum=True,
        fraction=0.2,
        chunks=512,
        seed=10,
        overwrite=True,
    )
    assert "table_pixels" in sdata_multi_c.tables
    assert sdata_multi_c.tables["table_pixels"].shape == (52088, 4)
    assert np.allclose(
        sdata_multi_c["table_pixels"].var["raw_image_post_norm_percentile_99.9"].values,
        np.array([0.9852, 0.9910, 0.9245, 0.9970]),
        atol=0.0001,
    )


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

    assert np.array_equal(result_array, correct_array, equal_nan=True)
