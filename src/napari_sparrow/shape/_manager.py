import itertools
from functools import singledispatchmethod
from typing import Any, Union

import dask
import geopandas
import numpy as np
import rasterio
import shapely
import spatialdata
from dask.array import Array
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.affinity import translate
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Translation

from napari_sparrow.image._image import _get_translation_values

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class ShapesLayerManager:
    def add_shapes(
        self,
        sdata: SpatialData,
        input: Union[Array, GeoDataFrame],
        output_layer: str,
        transformation: Union[Translation, Identity] = None,
        overwrite: bool = False,
    ) -> SpatialData:
        if transformation is not None and not isinstance(
            transformation, (Translation, Identity)
        ):
            raise ValueError(
                f"Currently only transformations of type Translation are supported, "
                f"while provided transformation is of type {type(transformation)}"
            )

        polygons = self.get_polygons_from_input(input)

        if transformation is not None:
            polygons = self.set_transformation(polygons, transformation)

        polygons = self.create_spatial_element(polygons)

        sdata = self.add_to_sdata(
            sdata,
            output_layer=output_layer,
            spatial_element=polygons,
            overwrite=overwrite,
        )

        return sdata

    def filter_shapes(
        self,
        sdata: SpatialData,
        indexes_to_keep: NDArray,
        prefix_filtered_shapes_layer: str,
    )->SpatialData:
        if len(indexes_to_keep) == 0:
            log.warning( f"Length of the 'indexes_to_keep' parameter is 0. "
                        "This would remove all shapes from sdata`. Skipping filtering step." )
            return sdata

        for _shapes_layer in [*sdata.shapes]:
            polygons = self.retrieve_data_from_sdata(sdata, name=_shapes_layer)
            polygons = self.get_polygons_from_input(polygons)

            current_indexes_shapes_layer = polygons.index.values.astype(int)

            bool_to_keep = np.isin(current_indexes_shapes_layer, indexes_to_keep)

            if sum(bool_to_keep) == 0:
                # no overlap, this means data.shapes[_shapes_layer] is
                # a polygons layer containing polygons filtered out in a previous step
                continue

            output_filtered_shapes_layer=f"{prefix_filtered_shapes_layer}_{_shapes_layer}"

            if sum( ~bool_to_keep ) == 0:
                # this is case where there are no polygons filtered out, so no
                # output_filtered_shapes_layer should be created
                log.warning(f"No polygons filtered out for shapes layer '{_shapes_layer}'. As a result, "
                            f"shapes layer '{output_filtered_shapes_layer}' will not be created. This is "
                            f"expected if 'indexes_to_keep' matches '{_shapes_layer}' indexes.")

                continue

            filtered_polygons = self.retrieve_data_from_sdata(
                sdata, name=_shapes_layer
            )[~bool_to_keep]

            log.info( f"Filtering {sum( ~bool_to_keep )} polygons from shapes layer '{_shapes_layer}'. "
                     f"Adding new shapes layer '{output_filtered_shapes_layer}' containing these filtered out polygons." )
            
            self.add_to_sdata(
                sdata,
                output_layer=output_filtered_shapes_layer,
                spatial_element=self.create_spatial_element(filtered_polygons),
                overwrite=True,
            )

            updated_polygons = self.retrieve_data_from_sdata(sdata, name=_shapes_layer)[
                bool_to_keep
            ]

            self.add_to_sdata(
                sdata,
                output_layer=_shapes_layer,
                spatial_element=self.create_spatial_element(updated_polygons),
                overwrite=True,
            )

        return sdata


    @singledispatchmethod
    def get_polygons_from_input(self, input: Any) -> GeoDataFrame:
        raise ValueError("Unsupported input type.")

    @get_polygons_from_input.register(Array)
    def _get_polygons_from_array(self, input: Array) -> GeoDataFrame:
        self.validate_input(input)
        return _mask_image_to_polygons(input)

    @get_polygons_from_input.register(GeoDataFrame)
    def _get_polygons_from_geodf(self, input: GeoDataFrame) -> GeoDataFrame:
        self.validate_input(input)
        return input

    @singledispatchmethod
    def validate_input(self, input: Any):
        raise ValueError("Unsupported input type.")

    @validate_input.register(Array)
    def _validate_array(self, input):
        assert (
            len(input.shape) == 2
        ), "Only 2D labels layers (y,x) are currently supported."

    @validate_input.register(GeoDataFrame)
    def _validate_gdf(self, input):
        has_z = input["geometry"].apply(lambda geom: geom.has_z)
        assert not has_z.any(), "Some polygons are not 2-dimensional!"

    def set_transformation(
        self, polygons: GeoDataFrame, transformation: Union[Translation, Identity]
    ) -> GeoDataFrame:
        x_translation, y_translation = _get_translation_values(transformation)

        polygons["geometry"] = polygons["geometry"].apply(
            lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
        )

        return polygons

    def create_spatial_element(
        self,
        polygons: GeoDataFrame,
    ) -> GeoDataFrame:
        return spatialdata.models.ShapesModel.parse(polygons)

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: GeoDataFrame,
        overwrite: bool = False,
    ) -> SpatialData:
        sdata.add_shapes(name=output_layer, shapes=spatial_element, overwrite=overwrite)
        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> GeoDataFrame:
        return sdata.shapes[name]

    def remove_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        del sdata.shapes[name]
        return sdata


def _mask_image_to_polygons(mask: Array) -> GeoDataFrame:
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
