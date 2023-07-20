from typing import Optional
import squidpy as sq
import matplotlib.pyplot as plt
import numpy as np

from napari_sparrow.table._table import _back_sdata_table_to_zarr


def nhood_enrichment(
    sdata, celltype_column: str = "annotation", output: Optional[str] = None
) -> None:
    """This function plots the nhood enrichment between different celltypes."""

    # remove 'nan' values from "adata.uns['annotation_nhood_enrichment']['zscore']"
    tmp = sdata.table.uns[f"{celltype_column}_nhood_enrichment"]["zscore"]
    sdata.table.uns[f"{celltype_column}_nhood_enrichment"]["zscore"] = np.nan_to_num(
        tmp
    )
    _back_sdata_table_to_zarr(sdata=sdata)
    sq.pl.nhood_enrichment(sdata.table, cluster_key=celltype_column, method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

