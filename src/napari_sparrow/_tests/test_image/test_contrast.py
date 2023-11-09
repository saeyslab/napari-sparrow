from spatialdata import SpatialData

from napari_sparrow.image._contrast import enhance_contrast


def test_enhance_contrast(sdata_multi_c):
    """
    Test enhance_contrast on 3D image with 2 channels.
    """
    sdata_multi_c = enhance_contrast(
        sdata_multi_c,
        img_layer="combine_z_16bit",
        output_layer="preprocessed_contrast",
        chunks=(2, 200, 200),
        overwrite=True,
    )

    assert "preprocessed_contrast" in sdata_multi_c.images
    assert isinstance(sdata_multi_c, SpatialData)
