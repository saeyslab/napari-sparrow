from typing import Dict, Tuple
import numpy as np
import spatialdata
from spatialdata import SpatialData

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def correct_marker_genes(
    sdata: SpatialData,
    celltype_correction_dict: Dict[str, Tuple[float, float]],
):
    """Returns the updated SpatialData object.

    Corrects celltypes that are higher expessed by dividing them by a value if they exceed a certain threshold.
    The celltype_correction_dict has as keys the celltypes that should be corrected and as values the threshold and the divider.
    """

    # Correct for all the genes
    for celltype, values in celltype_correction_dict.items():
        if celltype not in sdata.table.obs.columns:
            log.info(
                f"Cell type '{celltype}' not in obs of AnnData object. Skipping. Please first calculate gene expression for this cell type."
            )
            continue
        sdata.table.obs[celltype] = np.where(
            sdata.table.obs[celltype] < values[0],
            sdata.table.obs[celltype] / values[1],
            sdata.table.obs[celltype],
        )

    _back_sdata_table_to_zarr(sdata=sdata)

    return sdata


def filter_on_size(sdata: SpatialData, min_size: int = 100, max_size: int = 100000):
    """Returns the updated SpatialData object.

    All cells with a size outside of the min and max size range are removed.
    """

    start = sdata.table.shape[0]

    # Filter cells based on size and distance
    table = sdata.table[sdata.table.obs["shapeSize"] < max_size, :]
    table = table[table.obs["shapeSize"] > min_size, :]
    del sdata.table
    ## TODO: Look for a better way of doing this!
    sdata.table = spatialdata.models.TableModel.parse(table)

    sdata = _filter_shapes(sdata, filtered_name="size")

    filtered = start - table.shape[0]
    log.info( f"{filtered} cells were filtered out based on size." )

    return sdata


def _filter_shapes(sdata: SpatialData, filtered_name: str):
    for _shapes_layer in [*sdata.shapes]:
        if "filtered" not in _shapes_layer:
            #log.info(_shapes_layer)
            sdata[_shapes_layer].index = list(map(str, sdata[_shapes_layer].index))
            filtered_indexes = ~np.isin(
                sdata[_shapes_layer].index.values.astype(int),
                sdata.table.obs.index.values.astype(int),
            )

            if sum(filtered_indexes) != 0:
                sdata.add_shapes(
                    name=f"filtered_{_shapes_layer}_{filtered_name}",
                    shapes=spatialdata.models.ShapesModel.parse(
                        sdata[_shapes_layer][filtered_indexes]
                    ),
                    overwrite=True,
                )

            kept_indexes = np.isin(
                sdata[_shapes_layer].index.values.astype(int),
                sdata.table.obs.index.values.astype(int),
            )
            # we assume that sum(kept_indexes)!=0, i.e. that we did not filter all cells
            sdata.add_shapes(
                name=_shapes_layer,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[_shapes_layer][kept_indexes]
                ),
                overwrite=True,
            )

    return sdata


def _back_sdata_table_to_zarr(sdata: SpatialData):
    adata = sdata.table.copy()
    del sdata.table
    sdata.table = spatialdata.models.TableModel.parse(adata)
