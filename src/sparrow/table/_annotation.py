from __future__ import annotations

import os
from itertools import chain
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.table._keys import _ANNOTATION_KEY, _CLEANLINESS_KEY, _UNKNOWN_CELLTYPE_KEY
from sparrow.table._table import _back_sdata_table_to_zarr
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def score_genes(
    sdata: SpatialData,
    path_marker_genes: str | Path | pd.DataFrame,
    delimiter=",",
    row_norm: bool = False,
    repl_columns: dict[str, str] | None = None,
    del_celltypes: list[str] | None = None,
    input_dict=False,
    # TODO add annotation key here, let user pass it
    **kwargs: Any,
) -> tuple[dict, pd.DataFrame]:
    """
    The function loads marker genes from a CSV file and scores cells for each cell type using those markers using scanpy's `sc.tl.score_genes` function.

    Marker genes can be provided as a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column;
    or in dictionary format. The function further allows replacements of column names and deletions of specific marker genes.

    Parameters
    ----------
    sdata
        Data containing spatial information.
    path_marker_genes
        Path to the CSV file containing the marker genes or a pandas dataframe.
        It should be a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column.
    delimiter
        Delimiter used in the CSV file, default is ','.
    row_norm
        Flag to determine if row normalization is applied, default is False.
    repl_columns
        Dictionary containing cell types to be replaced. The keys are the original cell type names and
        the values are their replacements.
    del_celltypes
        List of cell types to be deleted from the list of possible cell type candidates.
        Cells are scored for these cell types, but will not be assigned a cell type from this list.
    input_dict
        If True, the marker gene list from the CSV file is treated as a dictionary with the first column being
        the cell type names and the subsequent columns being the marker genes for those cell types. Default is False.
    **kwargs
        Additional keyword arguments passed to `scanpy.tl.score_genes`.

    Returns
    -------
    A dictionary with cell types as keys and their respective marker genes as values.
    A DataFrame with the following structure. Index: cells, which corresponds to individual cell IDs. Columns: celltypes, as provided via the markers file. Values: Score obtained using scanpy's score_genes function for each cell type and for each cell.

    Notes
    -----
    The cell type `_UNKNOWN_CELLTYPE_KEY` is reserved for cells that could not be assigned a specific cell type.

    """
    # Load marker genes from csv
    if input_dict:
        log.warning(
            "'input_dict' is deprecated and will be removed in future versions. "
            "Please pass a pandas DataFrame or a path to a .csv file to 'path_marker_genes'. "
            "It should be a one-hot encoded matrix with cell types listed in the first row "
            "and marker genes in the first column."
        )
        df_markers = pd.read_csv(path_marker_genes, header=None, index_col=0, delimiter=delimiter)
        df_markers = df_markers.T
        genes_dict = df_markers.to_dict("list")
        for i in genes_dict:
            genes_dict[i] = [x for x in genes_dict[i] if str(x) != "nan"]
    # Replace column names in marker genes
    else:
        if isinstance(path_marker_genes, pd.DataFrame):
            df_markers = path_marker_genes
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
        _UNKNOWN_CELLTYPE_KEY not in genes_dict.keys()
    ), f"Cell type {_UNKNOWN_CELLTYPE_KEY} is reserved for cells that could not be assigned a specific cell type"

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(sdata.table, value, score_name=key, copy=False, **kwargs)
        except ValueError:
            log.warning(f"Markergenes {value} not present in region, celltype {key} not found")

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
        celltype_column=_ANNOTATION_KEY,
    )

    # add _UNKNOWN_CELLTYPE_KEY to the list of celltypes if it is detected.
    if _UNKNOWN_CELLTYPE_KEY in sdata.table.obs[_ANNOTATION_KEY].cat.categories:
        genes_dict[_UNKNOWN_CELLTYPE_KEY] = []

    _back_sdata_table_to_zarr(sdata)

    return genes_dict, scoresper_cluster


def score_genes_iter(
    sdata: SpatialData,
    path_marker_genes: str | Path | pd.DataFrame,
    delimiter: str = ",",
    norm_expr_var: bool = False,
    min_score: str | None = "Zero",
    min_score_q: int = 25,
    scaling="Nmarkers",
    scale_score_q=1,
    n_iter: int = 5,
    suffix="",
    output_dir=None,
) -> tuple[SpatialData, pd.DataFrame]:
    """
    Annotation algorithm.

    Parameters
    ----------
    sdata
        The SpatialData object.
    path_marker_genes
        Path to the CSV file containing the marker genes or a pandas dataframe.
        It should be a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column.
    delimiter
        Delimiter used in the CSV file.
    norm_expr_var
        Whether to normalize the gene expression matrix by the variance. Defaults to False, as we assume preprocessing is already employed.
    min_score
        Min score method. Choose from these options: "Zero", "Quantile", None.
    min_score_q
        Min score percentile. Ignored if `min_score` is not set to "Quantile".
    scaling
        Scaling method. Choose from these options: "MinMax", "ZeroMax", "Nmarkers", "Robust", "Rank".
    scale_score_q
        Scale score percentile.
    n_iter
        Number of iterations.
    suffix
        Suffix.
    output_dir
        If specified, figures with umaps will be saved in this directory after each iteration. If None, the plots will be displayed directly without saving.

    Returns
    -------
    An updated `sdata`.
    A DataFrame with the following structure. Index: cells, which corresponds to individual cell IDs. Columns: celltypes, as provided via the markers file. Values: Score obtained using sparrow's score_genes function for each cell type and for each cell.
    """
    adata = sdata.table

    kwargs = {}
    kwargs["norm_expr_var"] = norm_expr_var
    kwargs["min_score"] = min_score
    kwargs["min_score_q"] = min_score_q
    kwargs["scaling"] = scaling
    kwargs["scale_score_q"] = scale_score_q

    adata, df = _annotate_celltype_iter(
        adata=adata,
        path_marker_genes=path_marker_genes,
        delimiter=delimiter,
        n_iter=n_iter,
        suffix=suffix,
        output_dir=output_dir,
        **kwargs,  # keyword arguments passed to _annotate_celltype_weighted
    )

    _back_sdata_table_to_zarr(sdata)

    return sdata, df


def _annotate_celltype_iter(
    adata: AnnData,
    path_marker_genes: str | Path | pd.DataFrame,
    delimiter: str = ",",
    n_iter=5,
    suffix="",
    output_dir=None,
    **kwargs,
) -> tuple[AnnData, pd.DataFrame]:
    # initial clustering: = typical own_score_genes
    # 'mean expression' is over all cells'
    # but you do MinMax scaling so 'mean expression' does not have an effect

    if not isinstance(path_marker_genes, pd.DataFrame):
        marker_genes = pd.read_csv(path_marker_genes, index_col=0, delimiter=delimiter)
    elif isinstance(path_marker_genes, (str, Path)):
        marker_genes = path_marker_genes
    else:
        raise ValueError("Please pass either a path to a .csv file, or a pandas Dataframe to 'path_marker_genes'.")

    adata, scores = _annotate_celltype_weighted(
        adata,
        marker_genes=marker_genes,
        suffix=suffix,
        mean="all",
        mean_values=None,
        **kwargs,
    )

    log.info(
        (
            adata.obs["annotation_own_score_genes" + suffix].value_counts()
            / len(adata.obs["annotation_own_score_genes" + suffix])
        )
        * 100
    )

    sc.pl.umap(adata, color=["annotation_own_score_genes" + suffix], show=not output_dir)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"annotation_own_score_genes{suffix}"), bbox_inches="tight")
        plt.close()

    adata.obs["annotation_own_score_genes_start_iterative" + suffix] = adata.obs["annotation_own_score_genes" + suffix]

    adata.uns["own_score_genes_start_iterative" + suffix] = scores
    # iterative clustering:
    # own_score_genes again but now no (MinMax) scaling hence mean_expression has an effect
    # mean expression with fair contribution of each cell type (cell types are based on the previous clustering)
    changes = []
    completed = 0
    for _iteration in range(n_iter):
        log.info(f"Iteration {_iteration}.")
        cell_types = np.unique(adata.obs["annotation_own_score_genes" + suffix]).tolist()
        cell_types.remove(_UNKNOWN_CELLTYPE_KEY)
        mean_per_ct = []
        for ct in cell_types:
            l = pd.DataFrame(adata.obs["annotation_own_score_genes" + suffix] == ct)
            l = l.index[l["annotation_own_score_genes" + suffix]].tolist()
            ct_sel = adata[l, :]
            mean_per_ct.append(ct_sel.to_df().mean(axis=0))

        df = pd.concat(mean_per_ct, axis=1)
        next_mean = df.mean(axis=1)
        if "annotation_own_score_genes_previous" + suffix in adata.obs.columns:
            adata.obs.drop(columns=["annotation_own_score_genes_previous" + suffix], inplace=True)

        adata.obs.rename(
            columns={"annotation_own_score_genes" + suffix: "annotation_own_score_genes_previous" + suffix},
            inplace=True,
        )
        adata, scores = _annotate_celltype_weighted(
            adata,
            marker_genes=marker_genes,
            suffix=suffix,
            mean="given",
            mean_values=next_mean,
            **kwargs,
        )
        t = (
            adata.obs["annotation_own_score_genes" + suffix]
            == adata.obs["annotation_own_score_genes_previous" + suffix]
        )
        adata.obs["own_score_genes_diff_iter" + suffix] = [int(x) for x in t.to_list()]
        fr = adata.obs["own_score_genes_diff_iter" + suffix].value_counts() / len(
            adata.obs["own_score_genes_diff_iter" + suffix]
        )
        completed = completed + 1
        if len(fr) > 1 and (fr[0] * 100) > 0.05:
            log.info("Percentage of cells with changed annotation: " + str(np.round((fr[0] * 100), 2)))
            changes.append(fr[0] * 100)
            sc.pl.umap(adata, color=["own_score_genes_diff_iter" + suffix], show=not output_dir)
            if output_dir is not None:
                plt.savefig(
                    os.path.join(output_dir, f"own_score_genes_diff_iter{suffix}_{_iteration}"), bbox_inches="tight"
                )
                plt.close()
            sc.pl.umap(adata, color=["annotation_own_score_genes" + suffix], show=not output_dir)
            if output_dir is not None:
                plt.savefig(
                    os.path.join(output_dir, f"annotation_own_score_genes{suffix}_{_iteration}"), bbox_inches="tight"
                )
                plt.close()
            log.info(
                (
                    adata.obs["annotation_own_score_genes" + suffix].value_counts()
                    / len(adata.obs["annotation_own_score_genes" + suffix])
                )
                * 100
            )
        else:
            if len(fr) > 1:
                log.info("Percentage of cells with changed annotation: " + str(np.round((fr[0] * 100), 2)))
            else:
                log.info("Percentage of cells with changed annotation: " + str(0.0))
            log.info("converged")
            changes.append(0)
            break
    # plt.plot(list(range(1,completed+1,1)),changes)
    # make x-axis integers and start from 1
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # plt.xlabel('Iteration')
    # plt.ylabel('Percentage of cells with changed annotation')
    # save plot in folder output dir
    # if output_dir is not None:
    #    plt.savefig(os.path.join(output_dir, "own_score_genes_iterative_convergence" + suffix + ".png"))
    # drop columns from anndata.obs
    adata.obs.drop(columns=["own_score_genes_diff_iter" + suffix], inplace=True)
    adata.obs.drop(columns=["annotation_own_score_genes_previous" + suffix], inplace=True)

    scores.index = adata.obs.index

    return adata, scores


def _annotate_celltype_weighted(
    adata: AnnData,
    marker_genes: pd.DataFrame,
    norm_expr_var=False,
    min_score: str | None = "Zero",
    min_score_q=25,
    scaling="Nmarkers",
    scale_score_q=1,
    suffix="",
    mean: str = "all",
    mean_values=None,
) -> tuple[AnnData, pd.DataFrame]:
    # annotate each cell
    # method based on score_genes of scanpy but no bins and min max normalization of the scores per cell type
    # for each cell, a score is calculated for each cell type:
    # sum of the expressions of the markers in the cell - sum of the mean expressions of the markers in all cells
    # our expression data does not need to be scaled anymore (norm_expr_var = False) because sc.pp.scale is already applied in Sparrow
    # the input data should be normalized and scaled
    # create marker gene list
    # start time

    # TODO: check if everything ok with the anndata indices.

    _mean_options = (
        "all",
        "given",
    )
    if mean not in _mean_options:
        raise ValueError(f"'mean' should be one of {_mean_options}")
    _min_score_options = (
        "Zero",
        "Quantile",
        None,
    )
    if min_score not in _min_score_options:
        raise ValueError(f"'min_score' should be one of {_min_score_options}")
    _scaling_options = (
        "MinMax",
        "ZeroMax",
        "Nmarkers",
        "Robust",
        "Rank",
    )
    if scaling not in _scaling_options:
        raise ValueError(f"'scaling' should be one of {_scaling_options}")

    scores_cell_celltype = pd.DataFrame()
    cell_types = marker_genes.columns.tolist()
    # get the counts out
    matrix = adata.to_df()
    # correct for the variance of the expression of each gene

    # TODO: rewrite this part with mean == ...
    # the normalization for if it didn't happen yet
    if norm_expr_var:
        matrix = matrix.div(matrix.std(axis=0))
    if mean == "all":
        mean_expression = matrix.mean(axis=0)
    if mean == "given":
        mean_expression = mean_values

    # make sure the mean over all genes is zero, so no minus necessary anymore
    matrix_minus_mean = matrix - mean_expression
    genes_in_anndata = matrix.columns.to_list()
    # print time for the first part
    for cell_type in cell_types:
        adata.obs["score_" + cell_type] = 0
        for gene in marker_genes[marker_genes[cell_type] > 0].index.tolist():  # select marker genes per celltype
            if gene in genes_in_anndata:  # write
                adata.obs["score_" + cell_type] = (
                    adata.obs["score_" + cell_type] + matrix_minus_mean[gene]
                ) * marker_genes[cell_type][gene]

        scores_cell_celltype[cell_type] = adata.obs["score_" + cell_type]
        adata.obs = adata.obs.drop(columns=["score_" + cell_type])

    scores_cell_celltype.index.name = None
    scores_cell_celltype = scores_cell_celltype.reset_index(drop=True)

    # min score to obtain for a cell type, otherwise 'unknown'

    if min_score == "Zero":
        scores_cell_celltype_ok = scores_cell_celltype.copy(deep=True)
        # TODO rewrite, and use boolean arrays
        scores_cell_celltype_ok[scores_cell_celltype_ok > 0] = True
        # scores_cell_celltype_ok=scores_celltype_ok>0, just use the vbooleans
        scores_cell_celltype_ok[scores_cell_celltype_ok != True] = False  # noqa: E712 TODO
    if min_score == "Quantile":
        scores_cell_celltype_ok = scores_cell_celltype.copy(deep=True)
        scores_cell_celltype_ok[scores_cell_celltype_ok > scores_cell_celltype_ok.quantile(min_score_q / 100)] = True
        scores_cell_celltype_ok[scores_cell_celltype_ok != True] = False  # noqa: E712 TODO
    if min_score is None:
        scores_cell_celltype_ok = scores_cell_celltype.copy(deep=True)
        scores_cell_celltype_ok[scores_cell_celltype_ok.round(6) == scores_cell_celltype_ok.round(6)] = True

    # scale scores per cell type to make them more comparable between cell types (because some cell types have more markers etc.)
    # this scaling happens per celtype over the different cells
    if scaling == "MinMax":
        # if you chose this the '- mean_expression' you did before does not have an effect
        scores_cell_celltype = scores_cell_celltype.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    if scaling == "ZeroMax":
        scores_cell_celltype = scores_cell_celltype.apply(
            lambda x: (x) / (np.max(x))
        )  # (~ min max scaling with min = 0)

    if scaling == "Nmarkers":
        Nmarkers = marker_genes.sum(axis=0).to_list()
        scores_cell_celltype = scores_cell_celltype.div(Nmarkers)
        log.info("scaling based on number of markers per cell type")
    if scaling == "Robust":
        for cell_type in cell_types:
            if np.percentile(scores_cell_celltype[cell_type], scale_score_q) < np.percentile(
                scores_cell_celltype[cell_type], 100 - scale_score_q
            ):
                scores_cell_celltype[cell_type] = (
                    scores_cell_celltype[cell_type] - np.percentile(scores_cell_celltype[cell_type], scale_score_q)
                ) / (
                    np.percentile(scores_cell_celltype[cell_type], 100 - scale_score_q)
                    - np.percentile(scores_cell_celltype[cell_type], scale_score_q)
                )

            else:  # MinMax scaling if percentiles are equal
                scores_cell_celltype[cell_type] = (
                    scores_cell_celltype[cell_type] - np.min(scores_cell_celltype[cell_type])
                ) / (np.max(scores_cell_celltype[cell_type]) - np.min(scores_cell_celltype[cell_type]))

    if scaling == "Rank":
        for cell_type in cell_types:
            scores_cell_celltype[cell_type] = scores_cell_celltype[cell_type].rank(pct=True)

    # cell is annotated with the cell type with the highest score (+ this highest score is above min_score)
    to_return = scores_cell_celltype.copy(deep=True)
    scores_cell_celltype[scores_cell_celltype_ok == False] = 0  # np.nan  # noqa: E712 TODO
    # change the values of keys in list

    # cleanliness of each annotation is calculated
    # max_scores = scores_cell_celltype.max(axis=1)
    # second_scores = scores_cell_celltype.apply(lambda x: x.nlargest(2).values[-1], axis=1)
    max_scores, second_scores = (
        np.sort(scores_cell_celltype.values)[:, -1],
        np.sort(scores_cell_celltype.values)[:, -2:-1],
    )
    # make dataframes from max and second scores
    max_scores = pd.DataFrame(max_scores, index=scores_cell_celltype.index)
    second_scores = pd.DataFrame(second_scores, index=scores_cell_celltype.index)
    cleanliness = (max_scores - second_scores) / ((max_scores + second_scores + 0.0000001) / 2)
    # make cleanliness into a pd dataframe wxith cells as rows
    # cleanliness = pd.DataFrame(cleanliness, index=scores_cell_celltype.index)

    scores_cell_celltype[scores_cell_celltype_ok == False] = np.nan  # noqa: E712 TODO
    sc_cell_cellt = scores_cell_celltype.idxmax(axis=1).to_dict()

    unknown_cells = [k for k, v in sc_cell_cellt.items() if pd.isnull(v)]

    for i in unknown_cells:
        sc_cell_cellt[i] = _UNKNOWN_CELLTYPE_KEY
    sc_cell_cellt = {str(k): v for k, v in sc_cell_cellt.items()}
    adata.obs["annotation_own_score_genes" + suffix] = sc_cell_cellt.values()

    adata.obs["score_celltype_own_score_genes" + suffix] = max_scores.values
    adata.obs["second_score_celltype_own_score_genes" + suffix] = second_scores.values
    adata.obs["cleanliness_own_score_genes" + suffix] = cleanliness.values
    adata.uns["own_score_genes" + suffix] = scores_cell_celltype

    return adata, to_return


def cluster_cleanliness(
    sdata: SpatialData,
    celltypes: list[str],
    celltype_indexes: dict[str, int] | None = None,
    colors: list[str] | None = None,
) -> tuple[SpatialData, dict | None]:
    """
    Re-calculates annotations, potentially following corrections to the list of celltypes, or after a manual update of the assigned scores per cell type via e.g. `correct_marker_genes`.

    Celltypes can also be grouped together via the celltype_indexes parameter.
    Returns a `SpatialData` object alongside a dictionary mapping cell types to colors.

    Parameters
    ----------
    sdata
        Data containing spatial information.
    celltypes
        List of celltypes used for annotation.
    celltype_indexes
        Dictionary with cell type as keys and indexes as values.
        Cell types with provided indexes will be grouped together under new cell type provided as key.
        E.g.:
        celltype_indexes = {"fibroblast": [4,5,23,25], "stellate": [28,29,30]} ->
        celltypes at index 4,5,23 and 25 in provided list of celltypes (after an alphabetic sort) will be grouped together as "fibroblast".
    colors
        List of colors to be used for visualizing different cell types. If not provided,
        a default colormap will be generated.

    Returns
    -------
    Updated spatial data after the cleanliness analysis.
    Dictionary with cell types as keys and their corresponding colors as values.
    """
    celltypes = np.array(sorted(celltypes), dtype=str)
    color_dict = None

    # recalculate annotation, because we possibly did correction on celltype score for certain cells via correct_marker_genes function,
    # or updated the list of celltypes.
    sdata, _ = _annotate_celltype(
        sdata=sdata,
        celltypes=celltypes,
        row_norm=False,
        celltype_column=_ANNOTATION_KEY,
    )

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [mpl.colors.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)]

    sdata.table.uns[f"{_ANNOTATION_KEY}_colors"] = colors

    if celltype_indexes:
        sdata.table.obs[f"{_ANNOTATION_KEY}Save"] = sdata.table.obs[_ANNOTATION_KEY]
        gene_celltypes = {}

        for key, value in celltype_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, _indexes in celltype_indexes.items():
            sdata = _annotate_maxscore(gene, gene_celltypes, sdata)

        for gene, _indexes in celltype_indexes.items():
            sdata = _remove_celltypes(gene, gene_celltypes, sdata)

        celltypes_f = np.delete(celltypes, list(chain(*celltype_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(celltype_indexes.keys()))
        color_dict = dict(zip(celltypes_f, sdata.table.uns[f"{_ANNOTATION_KEY}_colors"]))

    else:
        color_dict = dict(zip(celltypes, sdata.table.uns[f"{_ANNOTATION_KEY}_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    sdata.table.uns[f"{_ANNOTATION_KEY}_colors"] = list(
        map(color_dict.get, sdata.table.obs[_ANNOTATION_KEY].cat.categories.values)
    )

    _back_sdata_table_to_zarr(sdata)

    return sdata, color_dict


def _annotate_celltype(
    sdata: SpatialData,
    celltypes: list[str],
    row_norm: bool = False,
    celltype_column: str = _ANNOTATION_KEY,
) -> tuple[SpatialData, pd.DataFrame]:
    scoresper_cluster = sdata.table.obs[[col for col in sdata.table.obs if col in celltypes]]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(scoresper_cluster.mean(axis=1).values, axis="rows").div(
            scoresper_cluster.std(axis=1).values, axis="rows"
        )
        sdata.table.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    sdata.table.obs[_CLEANLINESS_KEY] = scores.values

    def assign_cell_type(row):
        # Identify the cell type with the max score
        max_score_type = row.idxmax()
        # If max score is <= 0, assign _UNKNOWN_CELLTYPE_KEY
        if row[max_score_type] <= 0:
            return _UNKNOWN_CELLTYPE_KEY
        else:
            return max_score_type

    # Assign _UNKNOWN_CELLTYPE_KEY cell_type if no cell type could be found that has larger expression than random sample
    # as calculated by sc.tl.score_genes function of scanpy.
    sdata.table.obs[celltype_column] = scoresper_cluster.apply(assign_cell_type, axis=1)
    sdata.table.obs[celltype_column] = sdata.table.obs[celltype_column].astype("category")
    # Set the Cleanliness score for UNKNOWN_CELLTYPE_KEY equal to 0 (i.e. not clean)
    sdata.table.obs.loc[sdata.table.obs[celltype_column] == _UNKNOWN_CELLTYPE_KEY, _CLEANLINESS_KEY] = 0

    return sdata, scoresper_cluster


def _remove_celltypes(types: str, indexes: dict, sdata):
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in sdata.table.obs[_ANNOTATION_KEY].cat.categories:
            sdata.table.obs[_ANNOTATION_KEY] = sdata.table.obs[_ANNOTATION_KEY].cat.remove_categories(index)
    return sdata


def _annotate_maxscore(types: str, indexes: dict, sdata):
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    sdata.table.obs[_ANNOTATION_KEY] = sdata.table.obs[_ANNOTATION_KEY].cat.add_categories([types])
    for i, val in enumerate(sdata.table.obs[_ANNOTATION_KEY]):
        if val in indexes[types]:
            sdata.table.obs[_ANNOTATION_KEY][i] = types
    return sdata
