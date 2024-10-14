import re

import dask.array as da
import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import regionprops_table
from xrspatial import zonal_stats

from sparrow.utils._aggregate import RasterAggregator, _get_mask_area
from sparrow.utils._keys import _CELLSIZE_KEY


def test_aggregate_sum_dask_array():
    chunk_size = (2, 2)

    mask_dask_array = da.from_array(
        np.array([[3, 0, 0, 0], [0, 3, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]]),
        chunks=chunk_size,
    )
    mask_dask_array = mask_dask_array[None, ...]

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

    float_dask_array = float_dask_array[None, None, ...]

    aggregator = RasterAggregator(mask_dask_array=mask_dask_array, image_dask_array=float_dask_array)

    df_sum = aggregator.aggregate_sum()

    expected_result = np.array([51.5, 70.5, 6.0])

    assert np.array_equal(df_sum[0].values, expected_result)


def test_aggregate_sum(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_sum = aggregator.aggregate_sum()

    assert df_sum.shape[1] - 1 == image.shape[0]

    # check if we get same result as scipy

    scipy_sum = ndimage.sum_labels(input=image[0].compute(), labels=mask.compute(), index=da.unique(mask).compute())

    assert np.allclose(df_sum[0].values, scipy_sum, rtol=0, atol=1e-5)

    scipy_sum = ndimage.sum_labels(input=image[2].compute(), labels=mask.compute(), index=da.unique(mask).compute())

    assert np.allclose(df_sum[2].values, scipy_sum, rtol=0, atol=1e-5)

    # check if we get same results as xrspatial

    xrspatial_sum = zonal_stats(
        values=se_image[0],
        zones=se_labels,
        stats_funcs=["sum"],
    )

    xrspatial_sum = xrspatial_sum.compute()

    assert np.allclose(df_sum[0].values, xrspatial_sum["sum"].values, rtol=0, atol=1e-5)


def test_aggregate_min_max(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_max = aggregator.aggregate_max()

    assert df_max.shape[1] - 1 == image.shape[0]

    scipy_max = ndimage.labeled_comprehension(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.max,
        out_dtype=image.dtype,
        default=-np.inf,
    )

    assert np.allclose(df_max[0].values, scipy_max, rtol=0, atol=1e-5)

    df_min = aggregator.aggregate_min()

    assert df_min.shape[1] - 1 == image.shape[0]

    scipy_min = ndimage.labeled_comprehension(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.min,
        out_dtype=image.dtype,
        default=np.inf,
    )

    assert np.allclose(df_min[2].values, scipy_min, rtol=0, atol=1e-5)


def test_aggregate_mean(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_mean = aggregator.aggregate_mean()

    assert df_mean.shape[1] - 1 == image.shape[0]

    scipy_mean = ndimage.labeled_comprehension(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.mean,
        out_dtype=image.dtype,
        default=0,
    )

    assert np.allclose(df_mean[0].values, scipy_mean, rtol=0, atol=1e-5)

    scipy_mean = ndimage.labeled_comprehension(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.mean,
        out_dtype=image.dtype,
        default=0,
    )

    assert np.allclose(df_mean[2].values, scipy_mean, rtol=0, atol=1e-5)


def test_aggregate_var(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_var = aggregator.aggregate_var()

    assert df_var.shape[1] - 1 == image.shape[0]

    scipy_var = ndimage.variance(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[0].values, scipy_var, rtol=0, atol=1e-5)

    scipy_var = ndimage.variance(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[2].values, scipy_var, rtol=0, atol=1e-5)


def test_aggregate_var_3D(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    # make artificial 3D mask and image
    mask = da.concatenate([mask, mask + mask])
    image = da.concatenate(
        [
            image,
            image + image,
        ],
        axis=1,
    )

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk((1, 512, 512)),
        image_dask_array=image.rechunk((1, 1, 512, 512)),
    )
    df_var = aggregator.aggregate_var()

    assert df_var.shape[1] - 1 == image.shape[0]

    scipy_var = ndimage.variance(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[0].values, scipy_var, rtol=0, atol=1e-5)

    scipy_var = ndimage.variance(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[2].values, scipy_var, rtol=0, atol=1e-5)


def test_get_mask_area(sdata):
    se_labels = sdata["blobs_labels"]
    mask = se_labels.data[None, ...].rechunk(512)
    df = _get_mask_area(mask)

    mask_compute = mask.compute()
    area = ndimage.sum_labels(input=np.ones(mask_compute.shape), labels=mask_compute, index=np.unique(mask_compute))

    assert np.array_equal(df[_CELLSIZE_KEY].values, area)


def test_get_mask_area_subset(sdata):
    se_labels = sdata["blobs_labels"]
    mask = se_labels.data[None, ...].rechunk(512)
    mask_compute = mask.compute()

    index = np.unique(mask_compute)
    subset_index = [
        index[4],
        index[2],
        index.max() + 10,
    ]  # pick subset + an index not in mask. It will return 0 for this index.
    df = _get_mask_area(mask, index=subset_index)

    area = ndimage.sum_labels(input=np.ones(mask_compute.shape), labels=mask_compute, index=subset_index)

    assert np.array_equal(df[_CELLSIZE_KEY].values, area)


def test_aggregate_custom_channel(sdata_multi_c_no_backed):
    def _calculate_intensity_mean(mask_block, image_block):
        table = regionprops_table(label_image=mask_block, intensity_image=image_block, properties=["intensity_mean"])
        return table["intensity_mean"]

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    intensity_mean = aggregator._aggregate_custom_channel(
        image=image[0].rechunk(210),
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_intensity_mean,
    )

    intensity_mean_skimage = _calculate_intensity_mean(mask.compute(), image[0].compute()).astype(np.float32)

    assert np.array_equal(intensity_mean.flatten(), intensity_mean_skimage.flatten())


def test_aggregate_custom_channel_fails(sdata_multi_c_no_backed):
    def _calculate_centroid_weighted(mask_block, image_block):
        table = regionprops_table(label_image=mask_block, intensity_image=image_block, properties=["centroid_weighted"])
        return table["centroid_weighted-1"]

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    centroid = aggregator._aggregate_custom_channel(
        image=image[0].rechunk(210),
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_centroid_weighted,
    )

    centroid_skimage = _calculate_centroid_weighted(mask.compute(), image[0].compute()).astype(np.float32)

    # Does not work for calculation of centroid, because this requires global information.
    # could work if we would also pass block info to custom callable fn.
    with pytest.raises(AssertionError):
        assert np.array_equal(centroid.flatten(), centroid_skimage.flatten())


def test_aggregate_custom_channel_mask(sdata_multi_c_no_backed):
    # also works for euler,..etc
    def _calculate_eccentricity(mask_block):
        table = regionprops_table(label_image=mask_block[0], intensity_image=None, properties=["eccentricity"])
        return table["eccentricity"]

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )
    eccentricity = aggregator._aggregate_custom_channel(
        image=None,
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_eccentricity,
        dtype=np.float32,
    )
    eccentricity_skimage = _calculate_eccentricity(mask.compute()).astype(np.float32)

    assert np.array_equal(eccentricity.flatten(), eccentricity_skimage.flatten())


def test_aggregate_custom_channel_multiple_features(sdata_multi_c_no_backed):
    def _calculate_intensity_mean_area(mask_block, image_block):
        table = regionprops_table(
            label_image=mask_block, intensity_image=image_block, properties=["area", "intensity_mean"]
        )
        return np.stack([table["intensity_mean"], table["area"]], axis=1)

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    intensity_mean_area = aggregator._aggregate_custom_channel(
        image=image[0].rechunk(210),
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_intensity_mean_area,
        features=2,
    )

    intensity_mean_area_skimage = _calculate_intensity_mean_area(mask.compute(), image[0].compute()).astype(np.float32)

    assert np.array_equal(intensity_mean_area.reshape(-1, 2), intensity_mean_area_skimage.reshape(-1, 2))


def test_aggregate_custom_channel_multiple_features_sdata(sdata):
    def _calculate_intensity_mean_area(mask_block, image_block):
        table = regionprops_table(
            label_image=mask_block, intensity_image=image_block, properties=["area", "intensity_mean"]
        )
        return np.stack([table["intensity_mean"], table["area"]], axis=1)

    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    # sdata contains masks that span almost complete tissue, so an assertion error will be raised,
    # because we have masks with greater diameter than the depth.
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "We expect exactly one non-NaN element per row (each column corresponding to a chunk of 'mask'). Please consider increasing 'depth' parameter."
        ),
    ):
        aggregator._aggregate_custom_channel(
            image=image[0].rechunk(512),
            mask=mask.rechunk(512),
            depth=200,
            fn=_calculate_intensity_mean_area,
            features=2,
        )
