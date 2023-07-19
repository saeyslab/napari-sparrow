from typing import Optional
from spatialdata import SpatialData
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import scanpy as sc
import numpy as np

from napari_sparrow.plot._plot import plot_shapes
from napari_sparrow.image._image import _get_image_boundary


def scoreGenesPlot(
    sdata: SpatialData,
    scoresper_cluster: pd.DataFrame,
    img_layer: Optional[str] = None,
    shapes_layer: str = "segmentation_mask_boundaries",
    crd=None,
    filter_index: Optional[int] = None,
    output: Optional[str] = None,
) -> None:
    """This function plots the cleanliness and the leiden score next to the annotation."""
    if img_layer is None:
        img_layer = [*sdata.images][-1]
    si = sdata.images[img_layer]

    if crd is None:
        crd = _get_image_boundary(si)

    # Custom colormap:
    colors = np.concatenate(
        (plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20)))
    )
    colors = [
        mpl.colors.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)
    ]

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(sdata.table, color=["Cleanliness", "annotation"], show=False)

    if output:
        plt.savefig(output + "_Cleanliness_annotation", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    sc.pl.umap(sdata.table, color=["leiden", "annotation"], show=False)

    if output:
        plt.savefig(output + "_leiden_annotation", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # Plot annotation and cleanliness columns of sdata.table (AnnData) object
    sdata.table.uns["annotation_colors"] = colors
    plot_shapes(
        sdata=sdata,
        column="annotation",
        crd=crd,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        output=output + "_annotation" if output else None,
    )

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        sdata.table,
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
        show=False,
    )

    if output:
        plt.savefig(output + "_leiden_heatmap", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    if filter_index:
        sc.pl.heatmap(
            sdata.table[
                sdata.table.obs.leiden.isin(
                    [
                        str(index)
                        for index in range(filter_index, len(sdata.table.obs.leiden))
                    ]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
            show=False,
        )

        if output:
            plt.savefig(
                output + f"_leiden_heatmap_filtered_{filter_index}",
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()
