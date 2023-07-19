from typing import Optional
import scanpy as sc
import matplotlib.pyplot as plt
from spatialdata import SpatialData


def clustering_plot(sdata: SpatialData, output: Optional[str] = None) -> None:
    """This function plots the clusters and genes ranking"""

    # Leiden clustering
    sc.pl.umap(sdata.table, color=["leiden"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "_umap.png", bbox_inches="tight")
        plt.close()
        sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False, show=False)
        plt.savefig(output + "_rank_genes_groups.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False)
