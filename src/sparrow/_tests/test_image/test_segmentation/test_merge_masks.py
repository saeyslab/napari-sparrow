from spatialdata import SpatialData

from sparrow.image.segmentation._merge_masks import (
    merge_labels_layers,
    merge_labels_layers_nuclei,
)


def test_merge_labels_layers(sdata_multi_c: SpatialData):
    sdata_multi_c = merge_labels_layers(
        sdata_multi_c,
        labels_layer_1="masks_nuclear",
        labels_layer_2="masks_whole",
        output_labels_layer="masks_merged",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_merged" in sdata_multi_c.labels
    assert isinstance(sdata_multi_c, SpatialData)


def test_merge_labels_layers_nuclei(sdata_multi_c: SpatialData):
    sdata_multi_c = merge_labels_layers_nuclei(
        sdata_multi_c,
        labels_layer="masks_whole",
        labels_layer_nuclei_expanded="masks_nuclear",
        labels_layer_nuclei="masks_nuclear",
        output_labels_layer="masks_merged_nuclear",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_merged_nuclear" in sdata_multi_c.labels
    assert isinstance(sdata_multi_c, SpatialData)