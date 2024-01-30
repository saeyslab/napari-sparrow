import importlib.util

import pytest


@pytest.mark.skipif(not importlib.util.find_spec("sklearn"), reason="requires the scikit-learn library")
def test_sklearn(sdata_multi_c):
    from sklearn.cluster import KMeans

    X = sdata_multi_c.table.X
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    assert len(kmeans.labels_) == len(X)


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom(sdata_multi_c):
    from flowsom import FlowSOM

    adata = sdata_multi_c.table
    adata.var["channel"] = adata.var.index
    adata.var["marker"] = adata.var.index
    fsom = FlowSOM(adata, cols_to_use=[0, 1], xdim=10, ydim=10, n_clus=10)

    assert len(fsom.mudata) == len(adata)
