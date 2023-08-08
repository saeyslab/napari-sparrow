from typing import Optional
import scanpy as sc
import matplotlib.pyplot as plt
from spatialdata import SpatialData


def cluster(sdata: SpatialData, output: Optional[str] = None) -> None:
    """
    Plot the Leiden clusters on a UMAP, and show the most differentially expressed genes for each cluster on a second plot.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the analyzed data.
    output : str or None, optional
        The file path prefix for the plots (default is None).
        If provided, the plots will be saved to the specified output file path with "_umap.png" and "_rank_genes_groups.png" as suffixes.
        If None, the plots will be displayed directly without saving.

    Returns
    -------
    None
    """

    # Plot Leiden clusters on a UMAP
    sc.pl.umap(sdata.table, color=["leiden"], show=not output)
    if output:
        plt.savefig(output + "_umap.png", bbox_inches="tight")
        plt.close()

    # Plot the highly differential genes for each cluster
    sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False, show=False)
    if output:
        plt.savefig(output + "_rank_genes_groups.png", bbox_inches="tight")
        plt.close()
