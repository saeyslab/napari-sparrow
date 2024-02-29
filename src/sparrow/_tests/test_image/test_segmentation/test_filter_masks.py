import dask.array as da
from spatialdata import SpatialData

from sparrow.image.segmentation._filter_masks import (
    filter_labels_layer,
)


def test_merge_labels_layers(sdata_multi_c: SpatialData):
    sdata_multi_c = filter_labels_layer(
        sdata_multi_c,
        labels_layer="masks_whole",
        min_size=100,
        max_size=1000,
        depth=50,
        chunks=256,
        output_labels_layer="masks_whole_filtered",
        output_shapes_layer="masks_whole_filtered_boundaries",
        overwrite=True,
    )

    assert "masks_whole_filtered" in sdata_multi_c.labels
    assert (
        len(da.unique(sdata_multi_c.labels["masks_whole"].data).compute())
        - len(da.unique(sdata_multi_c.labels["masks_whole_filtered"].data).compute())
        == 55
    )
    assert isinstance(sdata_multi_c, SpatialData)
