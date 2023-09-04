import itertools
from typing import List, Optional

import dask
import dask.array as da
import geopandas
import numpy as np
import rasterio
import rasterio.features
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def _mask_image_to_polygons(mask: da.Array) -> geopandas.GeoDataFrame:
    """
    Convert a cell segmentation mask to polygons and return them as a GeoDataFrame.

    This function computes the polygonal outlines of the cells present in the
    given segmentation mask. The polygons are calculated in parallel using Dask
    delayed computations.

    Parameters
    ----------
    mask : dask.array.core.Array
        A Dask array representing the segmentation mask. Non-zero pixels belong
        to a cell; pixels with the same intensity value belong to the same cell.
        Zero pixels represent background (no cell).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing polygons extracted from the input mask. Each polygon
        is associated with a cell ID: the pixel intensity from the original mask.

    Notes
    -----
    The mask is processed in chunks to facilitate parallel computation. Polygons that
    are actually pieces of the same cell are combined back together to form coherent
    cells. This is necessary due to image chunking during parallel processing.

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da
    >>> from napari_sparrow.shape._shape import _mask_image_to_polygons
    >>> mask = da.from_array(np.array([[0, 3], [5, 5]]), chunks=(1, 1))
    >>> gdf = _mask_image_to_polygons(mask)
    >>> gdf
                                                    geometry
    cells
    3      POLYGON ((1.00000 0.00000, 1.00000 1.00000, 2....
    5      POLYGON ((0.00000 2.00000, 1.00000 2.00000, 2....
    >>> gdf.geometry[3]
    <POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))>
    >>> gdf.geometry[5]
    <POLYGON ((0 2, 1 2, 2 2, 2 1, 1 1, 0 1, 0 2))>
    """

    # Define a function to extract polygons and values from each chunk
    @dask.delayed
    def extract_polygons(mask_chunk: np.ndarray, chunk_coords: tuple) -> tuple:
        all_polygons = []
        all_values = []

        # Compute the boolean mask before passing it to the features.shapes() function
        bool_mask = mask_chunk > 0

        # Get chunk's top-left corner coordinates
        x_offset, y_offset = chunk_coords

        for shape, value in rasterio.features.shapes(
            mask_chunk.astype(np.int32),
            mask=bool_mask,
            transform=rasterio.Affine(1.0, 0, y_offset, 0, 1.0, x_offset),
        ):
            all_polygons.append(shapely.geometry.shape(shape))
            all_values.append(int(value))

        return all_polygons, all_values

    # Map the extract_polygons function to each chunk
    # Create a list of delayed objects

    chunk_coords = list(
        itertools.product(
            *[range(0, s, cs) for s, cs in zip(mask.shape, mask.chunksize)]
        )
    )

    delayed_results = [
        extract_polygons(chunk, coord)
        for chunk, coord in zip(mask.to_delayed().flatten(), chunk_coords)
    ]
    # Compute the results
    results = dask.compute(*delayed_results, scheduler="threads")

    # Combine the results into a single list of polygons and values
    all_polygons = []
    all_values = []
    for polygons, values in results:
        all_polygons.extend(polygons)
        all_values.extend(values)

    # Create a GeoDataFrame from the extracted polygons and values
    gdf = geopandas.GeoDataFrame({"geometry": all_polygons, "cells": all_values})

    # Combine polygons that are actually pieces of the same cell back together.
    # (These cells got broken into pieces because of image chunking, needed for parallel processing.)
    return gdf.dissolve(by="cells")


def _extract_boundaries_from_geometry_collection(geometry):
    if isinstance(geometry, Polygon):
        return [geometry.boundary]
    elif isinstance(geometry, MultiPolygon):
        return [polygon.boundary for polygon in geometry.geoms]
    elif isinstance(geometry, GeometryCollection):
        boundaries = []
        for geom in geometry.geoms:
            boundaries.extend(_extract_boundaries_from_geometry_collection(geom))
        return boundaries
    else:
        return []


def intersect_rectangles(
    rect1: List[int | float], rect2: List[int | float]
) -> Optional[List[int | float]]:
    """
    Calculate the intersection of two (axis aligned) rectangles.

    Parameters
    ----------
    rect1 : List[int | float]
        List representing the first rectangle [x_min, x_max, y_min, y_max].
    rect2 : List[int | float]
        List representing the second rectangle [x_min, x_max, y_min, y_max].

    Returns
    -------
    Optional[List[int | float]]
        List representing the intersection rectangle [x_min, x_max, y_min, y_max],
        or None if the rectangles do not overlap.
    """
    overlap_x = not (rect1[1] <= rect2[0] or rect2[1] <= rect1[0])
    overlap_y = not (rect1[3] <= rect2[2] or rect2[3] <= rect1[2])

    if overlap_x and overlap_y:
        x_min = max(rect1[0], rect2[0])
        x_max = min(rect1[1], rect2[1])
        y_min = max(rect1[2], rect2[2])
        y_max = min(rect1[3], rect2[3])
        return [x_min, x_max, y_min, y_max]
    else:
        return None
