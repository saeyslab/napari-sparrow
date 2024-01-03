from spatialdata import SpatialData
import scanpy as sc

from sparrow.table._table import _back_sdata_table_to_zarr


def cluster(
    sdata: SpatialData,
    pcs: int,
    neighbors: int,
    cluster_resolution: float = 0.8
) -> SpatialData:
    """
    Compute a neighborhood graph of the observations in a given SpatialData object,
    and a UMAP embedding, and perform Leiden clustering and differential
    gene analysis on the resulting clusters.

    Parameters
    ----------
    sdata : SpatialData
        Input SpatialData object containing spatial data and annotations.
    pcs : int
        Number of principal components for neighborhood graph construction.
    neighbors : int
        Number of neighbors for neighborhood graph construction.
    cluster_resolution : float, optional
        Resolution parameter for Leiden clustering (default is 0.8).

    Returns
    -------
    SpatialData
        The modified SpatialData object containing clustering results and embeddings.

    Notes
    -----
    - Computes neighborhood graph using `pcs` and `neighbors` parameters.
    - Calculates UMAP embedding based on the computed neighborhood graph.
    - Performs Leiden clustering with specified `cluster_resolution`.
    - Perform differential gene analysis on the Leiden clusters.
    - Saves the results back to the input SpatialData object.

    See also
    --------
    - pl.cluster : Visualize the clustering and differential gene expression.
    """

    # Neighborhood analysis
    sc.pp.neighbors(sdata.table, n_neighbors=neighbors, n_pcs=pcs, random_state=100)
    sc.tl.umap(sdata.table, random_state=100)

    # Leiden clustering
    sc.tl.leiden(sdata.table, resolution=cluster_resolution, random_state=100)
    sc.tl.rank_genes_groups(sdata.table, "leiden", method="wilcoxon")

    _back_sdata_table_to_zarr(sdata=sdata)

    return sdata
