from typing import List, Optional
from spatialdata import SpatialData
import matplotlib.pyplot as plt
import scanpy as sc

from napari_sparrow.plot._plot import plot_shapes


def cluster_cleanliness(
    sdata: SpatialData,
    shapes_layer: str = "segmentation_mask_boundaries",
    crd: Optional[List[int]] = None,
    color_dict: Optional[dict] = None,
    celltype_column: str = "annotation",
    output: Optional[str] = None,
) -> None:
    """This function plots the clustercleanliness as barplots, the images with colored celltypes and the clusters."""

    # Create the barplot
    stacked = (
        sdata.table.obs.groupby(["leiden", celltype_column], as_index=False)
        .size()
        .pivot("leiden", celltype_column)
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(sdata.table.obs.annotation.cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Use custom colormap
    if color_dict:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca(), color=color_dict)
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")

    # Save the barplot to ouput
    if output:
        fig.savefig(output + "_barplot.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    # Plot images with colored celltypes
    plot_shapes(
        sdata=sdata,
        column=celltype_column,
        alpha=0.8,
        shapes_layer=shapes_layer,
        output=output + f"_{celltype_column}" if output else None,
    )

    plot_shapes(
        sdata=sdata,
        column=celltype_column,
        crd=crd,
        alpha=0.8,
        shapes_layer=shapes_layer,
        output=output + f"_{celltype_column}_crop" if output else None,
    )

    # Plot clusters
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(
        sdata.table,
        color=[celltype_column],
        ax=ax,
        show=not output,
        size=300000 / sdata.table.shape[0],
    )
    ax.axis("off")

    if output:
        fig.savefig(output + f"_{celltype_column}_umap.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()
