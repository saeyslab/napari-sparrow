from typing import List, Optional
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import geopandas
import numpy as np
import shapely
import itertools
import rasterio
import rasterio.features
import dask.array as da
import dask


def _mask_image_to_polygons(mask: da.Array) -> geopandas.GeoDataFrame:
    """Returns the polygons as GeoDataFrame

    This function converts the mask to polygons.
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
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
    return geopandas.GeoDataFrame({"geometry": all_polygons, "cells": all_values})


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
    A: List[int | float], B: List[int | float]
) -> Optional[List[int | float]]:
    
    overlap_x = not (A[1] <= B[0] or B[1] <= A[0])
    overlap_y = not (A[3] <= B[2] or B[3] <= A[2])

    if overlap_x and overlap_y:
        # Calculate overlapping region
        x_min = max(A[0], B[0])
        x_max = min(A[1], B[1])
        y_min = max(A[2], B[2])
        y_max = min(A[3], B[3])

        # Return as a list: [x_min, x_max, y_min, y_max]
        return [x_min, x_max, y_min, y_max]
    else:
        return None
