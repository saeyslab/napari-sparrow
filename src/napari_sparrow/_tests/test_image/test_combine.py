from spatialdata import SpatialData

from napari_sparrow.image._combine import combine


def test_combine(sdata_multi_c: SpatialData):
    sdata_multi_c = combine(
        sdata_multi_c,
        img_layer="raw_image",
        output_layer="combine",
        nuc_channels=[15, 14],
        mem_channels=[0, 6, 11],
        overwrite=True,
    )

    assert "combine" in sdata_multi_c.images
    assert isinstance(sdata_multi_c, SpatialData)
