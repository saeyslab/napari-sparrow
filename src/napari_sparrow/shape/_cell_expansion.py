from shapely.geometry import Polygon
from longsgis import voronoiDiagram4plg
import geopandas
from spatialdata import SpatialData


def create_voronoi_boundaries(
    sdata: SpatialData,
    radius: int = 0,
    shapes_layer: str = "segmentation_mask_boundaries",
):
    if radius < 0:
        raise ValueError(
            f"radius should be >0, provided value for radius is '{radius}'"
        )

    sdata[shapes_layer].index = sdata[shapes_layer].index.astype("str")

    expanded_layer_name = "expanded_cells" + str(radius)
    # sdata[shape_layer].index = list(map(str, sdata[shape_layer].index))

    si = sdata[[*sdata.images][0]]

    # CHECKME: below we specify a boundary rectangle the size of the uncropped raw input image that was segmented.
    # Is that okay, or should we try to specify a "tight" boundary rectangle if the segmentation was only run on
    # crop of the image?

    margin = 200  # CHECKME: needed? why are margins not applied symmetrical along 4 sides of the boundary rectangle?
    boundary = Polygon(
        [
            (0, 0),
            (si.sizes["x"] + margin, 0),
            (si.sizes["x"] + margin, si.sizes["y"] + margin),
            (0, si.sizes["y"] + margin),
        ]
    )

    if expanded_layer_name in [*sdata.shapes]:
        del sdata.shapes[expanded_layer_name]
    sdata[expanded_layer_name] = sdata[shapes_layer].copy()
    sdata[expanded_layer_name]["geometry"] = sdata[shapes_layer].simplify(2)

    vd = voronoiDiagram4plg(sdata[expanded_layer_name], boundary)
    voronoi = geopandas.sjoin(
        vd, sdata[expanded_layer_name], predicate="contains", how="left"
    )
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
