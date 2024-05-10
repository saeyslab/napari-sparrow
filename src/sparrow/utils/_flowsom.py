from anndata import AnnData

from sparrow.utils._keys import _CLUSTERING_KEY, _METACLUSTERING_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import flowsom as fs
except ImportError:
    log.warning("'flowsom' not installed, 'sp.tb.flowsom' and `sp.im.flowsom` will not be available.")


def _flowsom(
    adata: AnnData,
    n_clusters: int = 10,
    **kwargs,
) -> tuple[AnnData, fs.FlowSOM]:
    fsom = fs.FlowSOM(adata, n_clusters=n_clusters, cols_to_use=None, **kwargs)
    if "cols_used" in adata.var:
        # can not back boolean column to zarr store
        adata.var["cols_used"] = adata.var["cols_used"].astype(int)
    # otherwise clusters can not be visualized in napari-spatialdata
    adata.obs[_METACLUSTERING_KEY] = adata.obs[_METACLUSTERING_KEY].astype("category")
    adata.obs[_CLUSTERING_KEY] = adata.obs[_CLUSTERING_KEY].astype("category")

    return adata, fsom
