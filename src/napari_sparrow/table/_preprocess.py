import scanpy as sc
from spatialdata import SpatialData

from napari_sparrow.shape._shape import _filter_shapes_layer
from napari_sparrow.table._table import _back_sdata_table_to_zarr
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def preprocess_anndata(
    sdata: SpatialData,
    shapes_layer: str = None,
    min_counts: int = 10,
    min_cells: int = 5,
    size_norm: bool = True,
    n_comps: int = 50,
) -> SpatialData:
    """
    Preprocess the table (AnnData) attribute of a SpatialData object. Filters cells and genes,
    normalizes based on nucleus/cell size, calculates QC metrics and principal components.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    shapes_layer : str, optional
        The shapes_layer of `sdata` that will be used to calculate nucleus size for normalization
        (or cell size if shapes_layer holds cell shapes).
        If not specified, the last shapes layer is chosen.
    min_counts : int, default=10
        Minimum number of genes a cell should contain to be kept.
    min_cells : int, default=5
        Minimum number of cells a gene should be in to be kept.
    size_norm : bool, default=True
        If True, normalization is based on the size of the nucleus/cell. Else the normalize_total function of scanpy is used.
    n_comps : int, default=50
        Number of principal components to calculate.

    Returns
    -------
    SpatialData
        The preprocessed `sdata` containg the preprocessed AnnData object as an attribute (sdata.table).

    Notes
    -----
    - All cells with less than `min_counts` genes and all genes with less than `min_cells` cells are removed.
    - The QC metrics are calculated using scanpy's `calculate_qc_metrics` function.

    Warnings
    --------
    - If the dimensionality of the table attribute is smaller than the desired number of principal components,
      `n_comps` is set to the minimum dimensionality and a message is printed.
    """

    # Calculate QC Metrics

    sc.pp.calculate_qc_metrics(sdata.table, inplace=True, percent_top=[2, 5])

    # Filter cells and genes
    sc.pp.filter_cells(sdata.table, min_counts=min_counts)
    sc.pp.filter_genes(sdata.table, min_cells=min_cells)

    # Normalize nucleus size
    if shapes_layer is None:
        shapes_layer = [*sdata.shapes][-1]
    sdata.table.obs["shapeSize"] = sdata[shapes_layer].area

    sdata.table.layers["raw_counts"] = sdata.table.X

    if size_norm:
        sdata.table.X = (sdata.table.X.T * 100 / sdata.table.obs.shapeSize.values).T
        sc.pp.log1p(sdata.table)
        # need to do .copy() here to set .raw value, because .scale still overwrites this .raw, which is unexpected behaviour
        sdata.table.raw = sdata.table.copy()
        sc.pp.scale(sdata.table, max_value=10)

    else:
        sc.pp.normalize_total(sdata.table)
        sc.pp.log1p(sdata.table)
        sdata.table.raw = sdata.table.copy()

    # calculate the max amount of pc's possible
    if min(sdata.table.shape) < n_comps:
        n_comps = min(sdata.table.shape)
        log.warning(
            (
                f"amount of pc's was set to {min( sdata.table.shape)} because of the dimensionality of the AnnData object."
            )
        )
    sc.tl.pca(sdata.table, svd_solver="arpack", n_comps=n_comps)

    # Is this the best way of doing it? Every time you subset your data, the polygons should be subsetted too!
    indexes_to_keep = sdata.table.obs.index.values.astype(int)
    sdata = _filter_shapes_layer(
        sdata,
        indexes_to_keep=indexes_to_keep,
        prefix_filtered_shapes_layer="filtered_low_counts",
    )

    # need to update sdata.table via .parse, otherwise it will not be backed by zarr store
    _back_sdata_table_to_zarr(sdata)

    return sdata
