from spatialdata import SpatialData

from sparrow.image.segmentation._expand_masks import expand_labels_layer


def test_expand_labels_layer(sdata_multi_c: SpatialData):
    sdata_multi_c = expand_labels_layer(
        sdata_multi_c,
        labels_layer="masks_cellpose_3D",
        distance=10,
        depth=100,
        chunks=256,
        output_labels_layer="masks_cellpose_3D_expanded",
        output_shapes_layer="masks_cellpose_3D_expanded_boundaries",
        overwrite=True,
    )

    assert "masks_cellpose_3D_expanded" in sdata_multi_c.labels
    assert "masks_cellpose_3D_expanded_boundaries" in sdata_multi_c.shapes

    assert isinstance(sdata_multi_c, SpatialData)
