from __future__ import annotations

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Any

import spatialdata
from dask.array import Array
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import BaseTransformation, set_transformation

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class LayerManager(ABC):
    def add_layer(
        self,
        sdata: SpatialData,
        arr: Array,
        output_layer: str,
        dims: tuple[str, ...] | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        transformation: BaseTransformation | dict[str, BaseTransformation] = None,
        scale_factors: ScaleFactors_t | None = None,
        overwrite: bool = False,
        **kwargs: Any,  # kwargs passed to create_spatial_element
    ) -> SpatialData:
        chunks = chunks or arr.chunksize
        if dims is None:
            log.warning("No dims parameter specified. Assuming order of dimension of provided array is (c, (z), y, x)")
            dims = self.get_dims(arr)

        intermediate_output_layer = None
        if scale_factors is not None:
            if sdata.is_backed():
                spatial_element = self.create_spatial_element(
                    arr,
                    dims=dims,
                    scale_factors=None,
                    chunks=chunks,
                    **kwargs,
                )
                if transformation is not None:
                    set_transformation(spatial_element, transformation)

                intermediate_output_layer = f"{uuid.uuid4()}_{output_layer}"
                log.info(f"Writing intermediate non-multiscale results to layer '{intermediate_output_layer}'")
                sdata = self.add_to_sdata(
                    sdata,
                    output_layer=intermediate_output_layer,
                    spatial_element=spatial_element,
                    overwrite=False,
                )
                arr = self.retrieve_data_from_sdata(sdata, intermediate_output_layer)
            else:
                arr = arr.persist()

        elif not sdata.is_backed():
            # if sdata is not backed, and if no scale factors, we also need to do a persist
            # to prevent recomputation
            arr = arr.persist()

        spatial_element = self.create_spatial_element(
            arr,
            dims=dims,
            scale_factors=scale_factors,
            chunks=chunks,
            **kwargs,
        )

        if transformation is not None:
            set_transformation(spatial_element, transformation)

        log.info(f"Writing results to layer '{output_layer}'")

        sdata = self.add_to_sdata(
            sdata,
            output_layer=output_layer,
            spatial_element=spatial_element,
            overwrite=overwrite,
        )

        if intermediate_output_layer:
            log.info(
                f"Removing intermediate output layer '{intermediate_output_layer}' from .zarr store at path {sdata.path}."
            )
            if os.path.isdir(sdata.path) and sdata.path.endswith(".zarr"):
                location = sdata._locate_spatial_element(sdata[intermediate_output_layer])

                shutil.rmtree(os.path.join(sdata.path, location[1], location[0]))
                sdata = self.remove_from_sdata(sdata, intermediate_output_layer)

        return sdata

    @abstractmethod
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[int, ...],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        **kwargs: Any,
    ) -> SpatialImage | MultiscaleSpatialImage:
        pass

    @abstractmethod
    def get_dims(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: SpatialImage | MultiscaleSpatialImage,
        overwrite: bool = False,
    ) -> SpatialData:
        pass

    @abstractmethod
    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        pass

    def remove_intermediate_layer(self, sdata: SpatialData, intermediate_output_layer: str) -> SpatialData:
        log.info(
            f"Removing intermediate output layer '{intermediate_output_layer}' from .zarr store at path {sdata.path}."
        )
        if os.path.isdir(sdata.path) and sdata.path.endswith(".zarr"):
            shutil.rmtree(os.path.join(sdata.path, "images", intermediate_output_layer))
            sdata = self.remove_from_sdata(sdata, intermediate_output_layer)
        return sdata

    @abstractmethod
    def remove_from_sdata(self, sdata, name):
        pass


class ImageLayerManager(LayerManager):
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[str, str, str],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, int, int] | int | None = None,
        c_coords: list[str] | None = None,
    ) -> SpatialImage | MultiscaleSpatialImage:
        if len(dims) == 3:
            return spatialdata.models.Image2DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                c_coords=c_coords,
            )
        elif len(dims) == 4:
            return spatialdata.models.Image3DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                c_coords=c_coords,
            )
        else:
            raise ValueError(
                f"Provided dims is {dims}, which is not supported, "
                "please provide dims parameter that only contains c, (z), y, and x."
            )

    def get_dims(self, arr) -> tuple[str, ...]:
        if len(arr.shape) == 3:
            return ("c", "y", "x")
        elif len(arr.shape) == 4:
            return ("c", "z", "y", "x")
        else:
            raise ValueError("Only 2D and 3D images (c, (z), y, x) are currently supported.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: SpatialImage | MultiscaleSpatialImage,
        overwrite: bool = False,
    ) -> SpatialData:
        sdata.add_image(name=output_layer, image=spatial_element, overwrite=overwrite)
        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.images[name].data

    def remove_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        del sdata.images[name]
        return sdata


class LabelLayerManager(LayerManager):
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[str, str],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, int] | int | None = None,
    ) -> SpatialImage | MultiscaleSpatialImage:
        if len(dims) == 2:
            return spatialdata.models.Labels2DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
            )
        elif len(dims) == 3:
            return spatialdata.models.Labels3DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
            )
        else:
            raise ValueError(
                f"Provided dims is {dims}, which is not supported, "
                "please provide dims parameter that only contains (z), y and x."
            )

    def get_dims(self, arr) -> tuple[str, str]:
        if len(arr.shape) == 2:
            return ("y", "x")
        elif len(arr.shape) == 3:
            return ("z", "y", "x")
        else:
            raise ValueError("Only 2D and 3D labels layers ( (z), y, x) are currently supported.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: SpatialImage | MultiscaleSpatialImage,
        overwrite: bool = False,
    ) -> SpatialData:
        sdata.add_labels(name=output_layer, labels=spatial_element, overwrite=overwrite)
        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.labels[name].data

    def remove_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        del sdata.labels[name]
        return sdata
