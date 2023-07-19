from typing import Tuple, Optional, List, Dict
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from spatialdata import SpatialData
import numpy as np
import pandas as pd
import scanpy as sc

from napari_sparrow.table._table import _back_sdata_table_to_zarr


def scoreGenes(
    sdata: SpatialData,
    path_marker_genes: str,
    delimiter=",",
    row_norm: bool = False,
    repl_columns: Optional[Dict[str, str]] = None,
    del_celltypes: Optional[List[str]] = None,
    input_dict=False,
) -> Tuple[dict, pd.DataFrame]:
    """Returns genes dict and the scores per cluster

    Load the marker genes from csv file in path_marker_genes.
    If the marker gene list is a one hot endoded matrix, leave the input as is.
    If the marker gene list is a Dictionary, with the first column the name of the celltype and the other columns the marker genes beloning to this celltype,
    input
    repl_columns holds the column names that should be replaced the in the marker genes.
    del_genes holds the marker genes that should be deleted from the marker genes and genes dict.
    """

    # Load marker genes from csv
    if input_dict:
        df_markers = pd.read_csv(
            path_marker_genes, header=None, index_col=0, delimiter=delimiter
        )
        df_markers = df_markers.T
        genes_dict = df_markers.to_dict("list")
        for i in genes_dict:
            genes_dict[i] = [x for x in genes_dict[i] if str(x) != "nan"]
    # Replace column names in marker genes
    else:
        df_markers = pd.read_csv(path_marker_genes, index_col=0, delimiter=delimiter)
        if repl_columns:
            for column, replace in repl_columns.items():
                df_markers.columns = df_markers.columns.str.replace(column, replace)

        # Create genes dict with all marker genes for every celltype
        genes_dict = {}
        for i in df_markers:
            genes = []
            for row, value in enumerate(df_markers[i]):
                if value > 0:
                    genes.append(df_markers.index[row])
            genes_dict[i] = genes

    assert (
        "unknown_celltype" not in genes_dict.keys()
    ), "Cell type 'unknown_celltype' is reserved for cells that could not be assigned a specific cell type"

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(sdata.table, value, score_name=key)
        except ValueError:
            warnings.warn(
                f"Markergenes {value} not present in region, celltype {key} not found"
            )

    # Delete genes from marker genes and genes dict
    if del_celltypes:
        for gene in del_celltypes:
            if gene in df_markers.columns:
                del df_markers[gene]
            if gene in genes_dict.keys():
                del genes_dict[gene]

    sdata, scoresper_cluster = _annotate_celltype(
        sdata=sdata,
        celltypes=df_markers.columns,
        row_norm=row_norm,
        celltype_column="annotation",
    )

    # add 'unknown_celltype' to the list of celltypes if it is detected.
    if "unknown_celltype" in sdata.table.obs["annotation"].cat.categories:
        genes_dict["unknown_celltype"] = []

    _back_sdata_table_to_zarr(sdata)

    return genes_dict, scoresper_cluster


def clustercleanliness(
    sdata: SpatialData,
    genes: List[str],
    gene_indexes: Optional[Dict[str, int]] = None,
    colors: Optional[List[str]] = None,
) -> Tuple[SpatialData, Optional[dict]]:
    """Returns a tuple with the AnnData object and the color dict."""

    celltypes = np.array(sorted(genes), dtype=str)
    color_dict = None

    # recalculate annotation, because we possibly did correction on celltype score for certain cells via correct_marker_genes function
    sdata, _ = _annotate_celltype(
        sdata=sdata,
        celltypes=celltypes,
        row_norm=False,
        celltype_column="annotation",
    )

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [
            mpl.colors.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)
        ]

    sdata.table.uns["annotation_colors"] = colors

    if gene_indexes:
        sdata.table.obs["annotationSave"] = sdata.table.obs.annotation
        gene_celltypes = {}

        for key, value in gene_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, indexes in gene_indexes.items():
            sdata = _annotate_maxscore(gene, gene_celltypes, sdata)

        for gene, indexes in gene_indexes.items():
            sdata = _remove_celltypes(gene, gene_celltypes, sdata)

        celltypes_f = np.delete(celltypes, list(chain(*gene_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(gene_indexes.keys()))
        color_dict = dict(zip(celltypes_f, sdata.table.uns["annotation_colors"]))

    else:
        color_dict = dict(zip(celltypes, sdata.table.uns["annotation_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    sdata.table.uns["annotation_colors"] = list(
        map(color_dict.get, sdata.table.obs.annotation.cat.categories.values)
    )

    _back_sdata_table_to_zarr(sdata)

    return sdata, color_dict


def _annotate_celltype(
    sdata: SpatialData,
    celltypes: List[str],
    row_norm: bool = False,
    celltype_column: str = "annotation",
) -> Tuple[SpatialData, pd.DataFrame]:
    scoresper_cluster = sdata.table.obs[
        [col for col in sdata.table.obs if col in celltypes]
    ]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(
            scoresper_cluster.mean(axis=1).values, axis="rows"
        ).div(scoresper_cluster.std(axis=1).values, axis="rows")
        sdata.table.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    sdata.table.obs["Cleanliness"] = scores.values

    def assign_cell_type(row):
        # Identify the cell type with the max score
        max_score_type = row.idxmax()
        # If max score is <= 0, assign 'unknown_celltype'
        if row[max_score_type] <= 0:
            return "unknown_celltype"
        else:
            return max_score_type

    # Assign 'unknown_celltype' cell_type if no cell type could be found that has larger expression than random sample
    # as calculated by sc.tl.score_genes function of scanpy.
    sdata.table.obs[celltype_column] = scoresper_cluster.apply(assign_cell_type, axis=1)
    sdata.table.obs[celltype_column] = sdata.table.obs[celltype_column].astype(
        "category"
    )
    # Set the Cleanliness score for unknown_celltype equal to 0 (i.e. not clean)
    sdata.table.obs.loc[
        sdata.table.obs[celltype_column] == "unknown_celltype", "Cleanliness"
    ] = 0

    return sdata, scoresper_cluster


def _remove_celltypes(types: str, indexes: dict, sdata):
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in sdata.table.obs.annotation.cat.categories:
            sdata.table.obs.annotation = (
                sdata.table.obs.annotation.cat.remove_categories(index)
            )
    return sdata


def _annotate_maxscore(types: str, indexes: dict, sdata):
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    sdata.table.obs.annotation = sdata.table.obs.annotation.cat.add_categories([types])
    for i, val in enumerate(sdata.table.obs.annotation):
        if val in indexes[types]:
            sdata.table.obs.annotation[i] = types
    return sdata
