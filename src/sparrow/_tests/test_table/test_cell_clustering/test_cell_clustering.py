import importlib

import dask.array as da
import pandas as pd
import pytest

from sparrow.table.cell_clustering._preprocess import cell_clustering_preprocess
from sparrow.utils._keys import _CLUSTERING_KEY, _METACLUSTERING_KEY


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_cell_clustering(sdata_blobs):
    """Integration test for cell clustering using flowsom"""
    import flowsom as fs

    from sparrow.image.pixel_clustering._clustering import flowsom as flowsom_pixel
    from sparrow.table.cell_clustering._clustering import flowsom as flowsom_cell

    img_layer = "blobs_image"
    labels_layer = "blobs_labels"
    table_layer = "table_cell_clustering"
    table_layer_flowsom = "table_cell_clustering_flowsom"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]
    fraction = 0.1

    sdata_blobs, _, _ = flowsom_pixel(
        sdata_blobs,
        img_layer=[img_layer],
        output_layer_clusters=[f"{img_layer}_clusters"],
        output_layer_metaclusters=[f"{img_layer}_metaclusters"],
        channels=channels,
        fraction=fraction,
        n_clusters=20,
        random_state=100,
        chunks=(1, 200, 200),
        overwrite=True,
    )

    sdata_blobs = cell_clustering_preprocess(
        sdata_blobs,
        labels_layer_cells=labels_layer,
        labels_layer_clusters=[f"{img_layer}_metaclusters"],
        output_layer=table_layer,
        chunks=(200, 200),
        overwrite=True,
    )

    assert table_layer in [*sdata_blobs.tables]

    # check if table_layer is of correct shape
    unique_labels = da.unique(sdata_blobs[labels_layer].data).compute()
    unique_labels = unique_labels[unique_labels != 0]

    unique_clusters = da.unique(sdata_blobs[f"{img_layer}_metaclusters"].data).compute()
    unique_clusters = unique_clusters[unique_clusters != 0]

    assert unique_labels.shape[0] == sdata_blobs.tables[table_layer].shape[0]
    assert unique_clusters.shape[0] == sdata_blobs.tables[table_layer].shape[1]

    sdata_blobs, fsom = flowsom_cell(
        sdata_blobs,
        labels_layer=labels_layer,
        table_layer=table_layer,
        output_layer=table_layer_flowsom,
        overwrite=True,
    )

    assert isinstance(fsom, fs.FlowSOM)
    # check that flowsom adds metaclusters and clusters to table
    assert _METACLUSTERING_KEY not in sdata_blobs.tables[table_layer].obs
    assert _CLUSTERING_KEY not in sdata_blobs.tables[table_layer].obs
    assert _METACLUSTERING_KEY in sdata_blobs.tables[table_layer_flowsom].obs
    assert _CLUSTERING_KEY in sdata_blobs.tables[table_layer_flowsom].obs

    # check that metacluster and cluster key are of categorical type, needed for visualization in napari-spatialdata
    assert isinstance(sdata_blobs[table_layer_flowsom].obs[_METACLUSTERING_KEY].dtype, pd.CategoricalDtype)
    assert isinstance(sdata_blobs[table_layer_flowsom].obs[_CLUSTERING_KEY].dtype, pd.CategoricalDtype)
