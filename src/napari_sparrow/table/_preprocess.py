import scanpy as sc

from napari_sparrow.table._table import _back_sdata_table_to_zarr, _filter_shapes

# TODO: add type hinting

def preprocess_anndata(
    sdata,
    nuc_size_norm: bool = True,
    n_comps: int = 50,
    min_counts=10,
    min_cells=5,
    shapes_layer=None,
):
    """Returns the new and original AnnData objects

    This function calculates the QC metrics.
    All cells with less than 10 genes and all genes with less than 5 cells are removed.
    Normalization is performed based on the size of the nucleus in nuc_size_norm."""
    # calculate the max amount of pc's possible
    if min(sdata.table.shape) < n_comps:
        n_comps = min(sdata.table.shape)
        print(
            "amount of pc's was set to " + str(min(sdata.table.shape)),
            " because of the dimensionality of the data.",
        )
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

    if nuc_size_norm:
        sdata.table.X = (sdata.table.X.T * 100 / sdata.table.obs.shapeSize.values).T
        sc.pp.log1p(sdata.table)
        # need to do .copy() here to set .raw value, because .scale still overwrites this .raw, which is unexpected behaviour
        sdata.table.raw = sdata.table.copy()
        sc.pp.scale(sdata.table, max_value=10)

    else:
        sc.pp.normalize_total(sdata.table)
        sc.pp.log1p(sdata.table)
        sdata.table.raw = sdata.table.copy()

    sc.tl.pca(sdata.table, svd_solver="arpack", n_comps=n_comps)
    # Is this the best way o doing it? Every time you subset your data, the polygons should be subsetted too!

    sdata = _filter_shapes(sdata, filtered_name="low_counts")

    # need to update sdata.table via .parse, otherwise it will not be backed by zarr store
    _back_sdata_table_to_zarr(sdata)

    return sdata
