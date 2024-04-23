import importlib.util

import pytest

from sparrow.table._clustering import kmeans, leiden
from sparrow.table._preprocess import preprocess_proteomics
from sparrow.table.pixel_clustering._create_pixel_matrix import create_pixel_matrix


@pytest.mark.skipif(not importlib.util.find_spec("sklearn"), reason="requires the scikit-learn library")
def test_sklearn(sdata_multi_c):
    from sklearn.cluster import KMeans

    X = sdata_multi_c.tables["table_intensities"].X
    result = KMeans(n_clusters=2, random_state=0).fit(X)
    assert len(result.labels_) == len(X)


def test_leiden(sdata_multi_c):
    assert "leiden" not in sdata_multi_c.tables["table_intensities"].obs.columns
    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        index_names_var=["0", "2", "5", "20"],
        random_state=100,
        overwrite=True,
    )
    assert "leiden" in sdata_multi_c.tables["table_intensities_clustered"].obs.columns
    assert sdata_multi_c["table_intensities_clustered"].shape == (674, 4)
    assert sdata_multi_c["table_intensities_clustered"].var.index.to_list() == ["0", "2", "5", "20"]

    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        index_positions_var=[0, 2, 5],
        random_state=100,
        overwrite=True,
    )
    assert "leiden" in sdata_multi_c.tables["table_intensities_clustered"].obs.columns
    assert sdata_multi_c["table_intensities_clustered"].shape == (674, 3)
    assert sdata_multi_c["table_intensities_clustered"].var.index.to_list() == ["0", "2", "5"]


def test_kmeans(sdata_multi_c):
    assert "kmeans" not in sdata_multi_c.tables["table_intensities"].obs.columns
    sdata_multi_c = kmeans(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="kmeans",
        random_state=100,
        overwrite=True,
    )
    assert "kmeans" in sdata_multi_c.tables["table_intensities_clustered"].obs.columns


def test_integration_clustering(sdata_multi_c):
    assert "leiden" not in sdata_multi_c.tables["table_intensities"].obs.columns

    sdata_multi_c = preprocess_proteomics(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities_preprocessed",
        output_layer="table_intensities_preprocessed",
        key_added="leiden",
        random_state=100,
        overwrite=True,
    )

    assert "leiden" in sdata_multi_c.tables["table_intensities_preprocessed"].obs.columns


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom_algo(sdata_multi_c):
    from flowsom import FlowSOM

    adata = sdata_multi_c.tables["table_intensities"]
    fsom = FlowSOM(adata, cols_to_use=[0, 1], xdim=10, ydim=10, n_clusters=10, seed=10)

    assert len(fsom.get_cell_data()) == len(adata)


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom(sdata_blobs):
    from flowsom import FlowSOM

    from sparrow.table._clustering import flowsom

    sdata_blobs = create_pixel_matrix(
        sdata_blobs,
        img_layer=["blobs_image"],
        output_img_layer=["blobs_image_preprocessed"],
        output_table_layer="table_pixels",
        channels=["lineage_0", "lineage_1", "lineage_5", "lineage_9"],
        q=99,
        q_sum=5,
        q_post=99.9,
        sigma=2.0,
        norm_sum=True,
        fraction=0.01,
        chunks=200,
        seed=10,
        overwrite=True,
    )

    sdata_blobs, fsom = flowsom(
        sdata_blobs,
        image_layer="blobs_image",
        table_layer="table_pixels",
        output_layer="table_pixels_flowsom",
        n_clusters=10,
        overwrite=True,
    )

    assert "table_pixels_flowsom" in sdata_blobs.tables
    assert sdata_blobs.tables["table_pixels_flowsom"].shape == sdata_blobs.tables["table_pixels"].shape
    assert sdata_blobs.tables["table_pixels_flowsom"].obs["flowsom"].unique().shape == (10,)
    assert isinstance(fsom, FlowSOM)
