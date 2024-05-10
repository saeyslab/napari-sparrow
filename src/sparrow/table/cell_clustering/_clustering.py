from __future__ import annotations

from typing import Iterable

from spatialdata import SpatialData

from sparrow.table._table import ProcessTable, _add_table_layer
from sparrow.utils._flowsom import _flowsom
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import flowsom as fs
except ImportError:
    log.warning("'flowsom' not installed, 'sp.tb.flowsom' will not be available.")


def flowsom(
    sdata: SpatialData,
    labels_layer: str | list[str] | None,
    table_layer: str,
    output_layer: str,
    n_clusters: int = 20,
    index_names_var: Iterable[str] | None = None,
    index_positions_var: Iterable[int] | None = None,
    random_state: int = 100,
    overwrite: bool = False,
    **kwargs,  # keyword arguments for _flowsom
) -> tuple[SpatialData, fs.FlowSOM]:
    """
    Executes the FlowSOM clustering algorithm on `table_layer` of the SpatialData object.

    This function applies the FlowSOM clustering algorithm (via `fs.FlowSOM`) on spatial data contained in a SpatialData object.
    The algorithm organizes data into self-organizing maps and then clusters these maps, grouping them into `n_clusters`.
    The results of this clustering are added to a new or existing table layer in the `sdata` object.

    Typically one would first process `sdata` via `sp.im.pixel_clustering_preprocess`, `sp.im.flowsom`, `sp.tb.cell_clustering_preprocess` before using this function.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    labels_layer : str | list[str] | None
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be clustered together (e.g. multiple samples).
    table_layer : str
        The table layer in `sdata` on which flowsom will be applied.
    output_layer : str
        The output table layer in `sdata` where results of the clustering and metaclustering will be stored.
    n_clusters : int, default=20
        The number of metaclusters to form from the self-organizing maps.
    index_names_var : Iterable[str] | None, optional
        Specifies the variable names to be used from `sdata.tables[table_layer].var` for clustering. If None, `index_positions_var` will be used if not None.
    index_positions_var : Iterable[int] | None, optional
        Specifies the positions of variables to be used from `sdata.tables[table_layer].var` for clustering. Used if `index_names_var` is None.
    random_state : int, default=100
        A random state for reproducibility of the clustering.
    overwrite : bool, default=False
        If True, overwrites the existing data in `output_layer` if it already exists.
    **kwargs
        Additional keyword arguments passed to the `fs.FlowSOM` clustering algorithm.

    Returns
    -------
    tuple[SpatialData, fs.FlowSOM]
        - The updated `sdata` with the clustering results added.
        - An instance of `fs.FlowSOM` containing the trained FlowSOM model.

    Warnings
    --------
    Ensure that the `table_layer` data is preprocessed appropriately to reflect the spatial and molecular features relevant for effective clustering.
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata(index_names_var=index_names_var, index_positions_var=index_positions_var)

    adata, fsom = _flowsom(
        adata,
        n_clusters=n_clusters,
        seed=random_state,
        **kwargs,
    )

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, fsom
