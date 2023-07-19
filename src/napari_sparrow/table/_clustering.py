from spatialdata import SpatialData
import scanpy as sc

from napari_sparrow.table._table import _back_sdata_table_to_zarr

# TODO: add type hints

def clustering(
    sdata: SpatialData, pcs: int, neighbors: int, cluster_resolution: float = 0.8
) -> SpatialData:
    """Returns the AnnData object.

    Performs neighborhood analysis, Leiden clustering and UMAP.
    Provides option to save the plots to output.
    """

    # Neighborhood analysis
    sc.pp.neighbors(sdata.table, n_neighbors=neighbors, n_pcs=pcs, random_state=100)
    sc.tl.umap(sdata.table, random_state=100)

    # Leiden clustering
    sc.tl.leiden(sdata.table, resolution=cluster_resolution, random_state=100)
    sc.tl.rank_genes_groups(sdata.table, "leiden", method="wilcoxon")

    _back_sdata_table_to_zarr(sdata=sdata)

    return sdata
