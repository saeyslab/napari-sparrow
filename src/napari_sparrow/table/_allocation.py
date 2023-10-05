from collections import namedtuple

import dask
import dask.dataframe as dd
import rasterio
import rasterio.features
import spatialdata
from affine import Affine
from anndata import AnnData
from spatialdata import SpatialData

from napari_sparrow.image._image import _get_spatial_element, _get_translation
from napari_sparrow.shape._shape import _filter_shapes_layer
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate(
    sdata: SpatialData,
    labels_layer: str = "segmentation_mask",
    shapes_layer: str = "segmentation_mask_boundaries",
    points_layer: str = "transcripts",
) -> SpatialData:
    """
    Allocates transcripts to cells via provided shapes_layer and points_layer and returns updated SpatialData
    augmented with a table attribute holding the AnnData object with cell counts.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    labels_layer : str, optional
        The layer in `sdata` that contains the masks corresponding to the shapes layer
        (possible before performing operation on the shapes layer, such as calculating voronoi expansion).
        Used for determining offset.
    shapes_layer : str, optional
        The layer in `sdata` that contains the boundaries of the segmentation mask, by default "segmentation_mask_boundaries".
    points_layer: str, optional
        The layer in `sdata` that contains the transcripts.

    Returns
    -------
        An updated SpatialData object with the added table attribute (AnnData object).
    """

    sdata[shapes_layer].index = sdata[shapes_layer].index.astype("str")

    # need to do this transformation,
    # because the polygons have same offset coords.x0 and coords.y0 as in segmentation_mask
    Coords = namedtuple("Coords", ["x0", "y0"])
    s_mask = _get_spatial_element(sdata, layer=labels_layer)
    coords = Coords(*_get_translation(s_mask))

    transform = Affine.translation(coords.x0, coords.y0)

    # Creating masks from polygons. TODO decide if you want to do this, even if voronoi is not calculated...
    # This is computationaly not heavy, but could take some ram,
    # because it creates image-size array of masks in memory
    # I guess not if no voronoi was created.
    log.info("Creating masks from polygons.")
    masks = rasterio.features.rasterize(
        zip(
            sdata[shapes_layer].geometry, sdata[shapes_layer].index.values.astype(float)
        ),
        out_shape=[s_mask.shape[0], s_mask.shape[1]],
        dtype="uint32",
        fill=0,
        transform=transform,
    )

    log.info(f"Created masks with shape {masks.shape}.")
    ddf = sdata[points_layer]

    log.info("Calculating cell counts.")

    # Define a function to process each partition using its index
    def process_partition(index, masks, coords):
        partition = ddf.get_partition(index).compute()

        filtered_partition = partition[
            (coords.y0 < partition["y"])
            & (partition["y"] < masks.shape[0] + coords.y0)
            & (coords.x0 < partition["x"])
            & (partition["x"] < masks.shape[1] + coords.x0)
        ]

        filtered_partition["cells"] = masks[
            filtered_partition["y"].values.astype(int) - int(coords.y0),
            filtered_partition["x"].values.astype(int) - int(coords.x0),
        ]

        return filtered_partition

    # Get the number of partitions in the Dask DataFrame
    num_partitions = ddf.npartitions

    # Process each partition using its index
    processed_partitions = [
        dask.delayed(process_partition)(i, masks, coords) for i in range(num_partitions)
    ]

    # Combine the processed partitions into a single DataFrame
    combined_partitions = dd.from_delayed(processed_partitions)

    coordinates = combined_partitions.groupby("cells").mean().iloc[:, [0, 1]]
    cell_counts = combined_partitions.groupby(["cells", "gene"]).size()

    coordinates, cell_counts = dask.compute(
        coordinates, cell_counts, scheduler="threads"
    )

    cell_counts = cell_counts.unstack(fill_value=0)

    log.info("Finished calculating cell counts.")

    # make sure coordinates are sorted in same order as cell_counts
    index_order = cell_counts.index.argsort()
    coordinates = coordinates.loc[cell_counts.index[index_order]]

    log.info("Creating AnnData object.")

    # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values

    adata.obs["region"] = 1
    adata.obs["instance"] = 1

    if sdata.table:
        del sdata.table

    sdata.table = spatialdata.models.TableModel.parse(
        adata, region_key="region", region=1, instance_key="instance"
    )

    indexes_to_keep = sdata.table.obs.index.values.astype(int)
    sdata = _filter_shapes_layer(
        sdata,
        indexes_to_keep=indexes_to_keep,
        prefix_filtered_shapes_layer="filtered_segmentation",
    )

    return sdata
