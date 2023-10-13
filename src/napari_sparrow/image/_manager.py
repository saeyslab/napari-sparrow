from __future__ import annotations

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import spatialdata
from dask.array import Array
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import BaseTransformation, set_transformation

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class LayerManager(ABC):
    def add_layer(
        self,
        sdata: SpatialData,
        arr: Array,
        output_layer: str,
        chunks: Optional[str | Tuple[int, ...] | int] = None,
        transformation: Union[BaseTransformation, dict[str, BaseTransformation]] = None,
        scale_factors: Optional[ScaleFactors_t] = None,
        overwrite: bool = False,
    ) -> SpatialData:
        self.arr = arr
        self.output_layer = output_layer
        self.chunks = chunks or arr.chunksize
        self.transformation = transformation
        self.scale_factors = scale_factors
        self.overwrite = overwrite

        self.validate_input()

        intermediate_output_layer = None
        if self.scale_factors:
            if sdata.is_backed():
                spatial_element = self.create_spatial_element(
                    self.arr,
                    dims=self.get_dims(),
                    scale_factors=None,
                    chunks=self.chunks,
                )
                if self.transformation:
                    set_transformation(spatial_element, self.transformation)

                intermediate_output_layer = f"{uuid.uuid4()}_{self.output_layer}"
                log.info(
                    f"Writing intermediate non-multiscale results to layer '{intermediate_output_layer}'"
                )
                sdata = self.add_to_sdata(
                    sdata,
                    output_layer=intermediate_output_layer,
                    spatial_element=spatial_element,
                    overwrite=False,
                )
                self.arr = self.retrieve_data_from_sdata(
                    sdata, intermediate_output_layer
                )
            else:
                self.arr = self.arr.persist()

        elif not sdata.is_backed():
            # if sdata is not backed, and if no scale factors, we also need to do a persist
            # to prevent recomputation
            self.arr = self.arr.persist()

        spatial_element = self.create_spatial_element(
            self.arr,
            dims=self.get_dims(),
            scale_factors=self.scale_factors,
            chunks=self.chunks,
        )

        if self.transformation:
            set_transformation(spatial_element, self.transformation)

        log.info(f"Writing results to layer '{self.output_layer}'")

        sdata = self.add_to_sdata(
            sdata,
            output_layer=self.output_layer,
            spatial_element=spatial_element,
            overwrite=self.overwrite,
        )

        if intermediate_output_layer:
            log.info(
                f"Removing intermediate output layer '{intermediate_output_layer}' from .zarr store at path {sdata.path}."
            )
            if os.path.isdir(sdata.path) and sdata.path.endswith(".zarr"):
                location = sdata._locate_spatial_element(
                    sdata[intermediate_output_layer]
                )

                shutil.rmtree(os.path.join(sdata.path, location[1], location[0]))
                sdata = self.remove_from_sdata(sdata, intermediate_output_layer)

        return sdata

    @abstractmethod
    def validate_input(self):
        pass

    @abstractmethod
    def create_spatial_element(
        self,
        arr: Array,
        dims: Tuple[int, ...],
        scale_factors: Optional[ScaleFactors_t] = None,
        chunks: Optional[str | Tuple[int, ...] | int] = None,
    ) -> Union[SpatialImage, MultiscaleSpatialImage]:
        pass

    @abstractmethod
    def get_dims(self) -> Tuple[str, ...]:
        pass

    @abstractmethod
    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: Union[SpatialImage, MultiscaleSpatialImage],
        overwrite: bool = False,
    ) -> SpatialData:
        pass

    @abstractmethod
    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        pass

    def remove_intermediate_layer(
        self, sdata: SpatialData, intermediate_output_layer: str
    ) -> SpatialData:
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
    def validate_input(self):
        assert (
            len(self.arr.shape) == 3
        ), "Only 2D images (c,y,x) are currently supported."

    def create_spatial_element(
        self,
        arr: Array,
        dims: Tuple[str, str, str],
        scale_factors: Optional[ScaleFactors_t] = None,
        chunks: Optional[str | Tuple[int, int, int] | int] = None,
    ) -> Union[SpatialImage, MultiscaleSpatialImage]:
        return spatialdata.models.Image2DModel.parse(
            arr, dims=dims, scale_factors=scale_factors, chunks=chunks
        )

    def get_dims(self) -> Tuple[str, str, str]:
        return ("c", "y", "x")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: Union[SpatialImage, MultiscaleSpatialImage],
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
    def validate_input(self):
        assert (
            len(self.arr.shape) == 2
        ), "Only 2D labels layers (y,x) are currently supported."

    def create_spatial_element(
        self,
        arr: Array,
        dims: Tuple[str, str],
        scale_factors: Optional[ScaleFactors_t] = None,
        chunks: Optional[str | Tuple[int, int] | int] = None,
    ) -> Union[SpatialImage, MultiscaleSpatialImage]:
        return spatialdata.models.Labels2DModel.parse(
            arr, dims=dims, scale_factors=scale_factors, chunks=chunks
        )

    def get_dims(self) -> Tuple[str, str]:
        return ("y", "x")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: Union[SpatialImage, MultiscaleSpatialImage],
        overwrite: bool = False,
    ) -> SpatialData:
        sdata.add_labels(name=output_layer, labels=spatial_element, overwrite=overwrite)
        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.labels[name].data

    def remove_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        del sdata.labels[name]
        return sdata
