from spatialdata import SpatialData

from sparrow.image._filters import gaussian_filtering, min_max_filtering


def test_min_max_filtering(sdata_multi_c: SpatialData):
    """
    Test min max filtering on 3D image with 2 channels.
    """
    sdata_multi_c = min_max_filtering(
        sdata_multi_c,
        img_layer="combine_z",
        output_layer="preprocessed_min_max",
        overwrite=True,
    )

    assert "preprocessed_min_max" in sdata_multi_c.images
    assert isinstance(sdata_multi_c, SpatialData)


def test_gaussian_filtering(sdata_multi_c: SpatialData):
    """
    Test gaussian filtering on 3D image with 2 channels.
    """
    sdata_multi_c = gaussian_filtering(
        sdata_multi_c,
        img_layer="combine_z",
        output_layer="preprocessed_gaussian",
        overwrite=True,
    )

    assert "preprocessed_gaussian" in sdata_multi_c.images
    assert isinstance(sdata_multi_c, SpatialData)
