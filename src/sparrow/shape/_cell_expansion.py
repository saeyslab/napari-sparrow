import geopandas
from longsgis import voronoiDiagram4plg
from shapely.geometry import Polygon
from spatialdata import SpatialData

from sparrow.image._image import _get_translation


def create_voronoi_boundaries(
    sdata: SpatialData,
    radius: int = 0,
    shapes_layer: str = "segmentation_mask_boundaries",
) -> SpatialData:
    """
    Create Voronoi boundaries from the shapes layer of the provided SpatialData object.

    Given spatial data and a radius, this function calculates Voronoi boundaries
    and expands these boundaries based on the radius.

    Parameters
    ----------
    sdata : SpatialData
        The spatial data object on which Voronoi boundaries will be created.
    radius : int, optional
        The expansion radius for the Voronoi boundaries, by default 0.
        If provided, Voronoi boundaries will be expanded by this radius.
        Must be non-negative.
    shapes_layer : str, optional
        The name of the layer in `sdata` representing shapes used to derive
        Voronoi boundaries. Default is "segmentation_mask_boundaries".

    Returns
    -------
    SpatialData
        Modified `sdata` object with the Voronoi boundaries created and
        possibly expanded.

    Raises
    ------
    ValueError
        If the provided radius is negative.
    """
    if radius < 0:
        raise ValueError(f"radius should be >0, provided value for radius is '{radius}'")

    sdata[shapes_layer].index = sdata[shapes_layer].index.astype("str")

    expanded_layer_name = "expanded_cells" + str(radius)
    # sdata[shape_layer].index = list(map(str, sdata[shape_layer].index))

    si = sdata[[*sdata.images][0]]
    # need to add translation in x and y direction to size of the image,
    # to account for use case where si is already cropped
    tx, ty = _get_translation(si)

    boundary = Polygon(
        [
            (tx, ty),
            (tx + si.sizes["x"], ty),
            (tx + si.sizes["x"], ty + si.sizes["y"]),
            (tx, ty + si.sizes["y"]),
        ]
    )

    if expanded_layer_name in [*sdata.shapes]:
        del sdata.shapes[expanded_layer_name]
    sdata[expanded_layer_name] = sdata[shapes_layer].copy()
    sdata[expanded_layer_name]["geometry"] = sdata[shapes_layer].simplify(2)

    vd = voronoiDiagram4plg(sdata[expanded_layer_name], boundary)
    voronoi = geopandas.sjoin(vd, sdata[expanded_layer_name], predicate="contains", how="left")
    voronoi.index = voronoi.index_right
    voronoi = voronoi[~voronoi.index.duplicated(keep="first")]
    voronoi = _delete_overlap(voronoi, sdata[expanded_layer_name])

    buffered = sdata[expanded_layer_name].buffer(distance=radius)
    intersected = voronoi.sort_index().intersection(buffered.sort_index())

    sdata[expanded_layer_name].geometry = intersected

    return sdata


def _delete_overlap(voronoi, polygons):
    I1, I2 = voronoi.sindex.query_bulk(voronoi["geometry"], predicate="overlaps")
    voronoi2 = voronoi.copy()

    for cell1, cell2 in zip(I1, I2):
        # if cell1!=cell2:
        voronoi.geometry.iloc[cell1] = voronoi.iloc[cell1].geometry.intersection(
            voronoi2.iloc[cell1].geometry.difference(voronoi2.iloc[cell2].geometry)
        )
        voronoi.geometry.iloc[cell2] = voronoi.iloc[cell2].geometry.intersection(
            voronoi2.iloc[cell2].geometry.difference(voronoi2.iloc[cell1].geometry)
        )
    voronoi["geometry"] = voronoi.geometry.union(polygons.geometry)
    polygons = polygons.buffer(distance=0)
    voronoi = voronoi.buffer(distance=0)
    return voronoi
