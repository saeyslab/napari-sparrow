from typing import Dict, Tuple
from collections import namedtuple
import numpy as np
import spatialdata
from spatialdata import SpatialData


def correct_marker_genes(
    sdata: SpatialData,
    celltype_correction_dict: Dict[str, Tuple[float, float]],
):
    """Returns the new AnnData object.

    Corrects marker genes that are higher expessed by dividing them.
    The genes has as keys the genes that should be corrected and as values the threshold and the divider.
    """

    # Correct for all the genes
    for celltype, values in celltype_correction_dict.items():
        if celltype not in sdata.table.obs.columns:
            print(
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
    """Returns a tuple with the AnnData object and the number of filtered cells.

    All cells outside of the min and max size range are removed.
    If the distance between the location of the transcript and the center of the polygon is large, the cell is deleted.
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
    print(str(filtered) + " cells were filtered out based on size.")

    return sdata


def _filter_shapes(sdata: SpatialData, filtered_name: str):
    for _shapes_layer in [*sdata.shapes]:
        if "filtered" not in _shapes_layer:
            print(_shapes_layer)
            sdata[_shapes_layer].index = list(map(str, sdata[_shapes_layer].index))
            filtered_indexes = ~np.isin(
                sdata[_shapes_layer].index.values.astype(int),
                sdata.table.obs.index.values.astype(int),
            )

            if sum(filtered_indexes) != 0:
                sdata.add_shapes(
                    name=f"filtered_{filtered_name}_{_shapes_layer}",
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