import numpy as np

from sparrow.table.pixel_clustering._create_pixel_matrix import create_pixel_matrix


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
    assert sdata_multi_c.tables["table_pixels"].shape == (52099, 4)
    assert sdata_multi_c.tables["table_pixels"].to_df().isna().any(axis=0).sum() == 0
    assert np.allclose(
        sdata_multi_c["table_pixels"].var["raw_image_post_norm_percentile_99.9"].values,
        np.array([0.9852, 0.9910, 0.9245, 0.9970]),
        atol=0.0001,
    )
