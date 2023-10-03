import warnings
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from dask.array import Array
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import SpatialElement
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import BaseTransformation
from spatialdata.transformations._utils import (
    _get_transformations,
    _get_transformations_xarray,
)
from xarray import DataArray

from napari_sparrow.utils.pylogger import get_pylogger
from napari_sparrow.image._manager import LabelLayerManager, ImageLayerManager

log = get_pylogger(__name__)


def _substract_translation_crd(
    spatial_image: Union[SpatialImage, DataArray],
    crd=Tuple[int, int, int, int],
) -> Tuple[int, int, int, int] | None:
    tx, ty = _get_translation(spatial_image)

    _crd = crd
    crd = [
        int(max(0, crd[0] - tx)),
        max(0, int(min(spatial_image.sizes["x"], crd[1] - tx))),
        int(max(0, crd[2] - ty)),
        max(0, int(min(spatial_image.sizes["y"], crd[3] - ty))),
    ]

    if crd[1] - crd[0] <= 0 or crd[3] - crd[2] <= 0:
        warnings.warn(
            f"Crop param {_crd} after correction for possible translation on "
            f"SpatialImage object '{spatial_image.name}' is "
            f"'{crd}. Falling back to setting crd to 'None'."
        )
        crd = None

    return crd


def _get_boundary(
    spatial_image: Union[SpatialImage, DataArray]
) -> Tuple[int, int, int, int]:
    tx, ty = _get_translation(spatial_image)
    width = spatial_image.sizes["x"]
    height = spatial_image.sizes["y"]
    return (int(tx), int(tx + width), int(ty), int(ty + height))


def _get_translation(
    spatial_image: Union[SpatialImage, MultiscaleSpatialImage, DataArray]
) -> Tuple[float, float]:
    transform_matrix = _get_transformation(spatial_image).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )

    if (
        transform_matrix[0, 0] == 1.0
        and transform_matrix[0, 1] == 0.0
        and transform_matrix[1, 0] == 0.0
        and transform_matrix[1, 1] == 1.0
        and transform_matrix[2, 0] == 0.0
        and transform_matrix[2, 1] == 0.0
        and transform_matrix[2, 2] == 1.0
    ):
        return transform_matrix[0, 2], transform_matrix[1, 2]
    else:
        raise ValueError(
            f"The provided transform matrix '{transform_matrix}' associated with the SpatialImage "
            f"element with name '{spatial_image.name}' represents more than just a translation, which is not currently supported."
        )


def _apply_transform(
    se: Union[SpatialImage, DataArray]
) -> Tuple[Union[SpatialImage, DataArray], np.ndarray, np.ndarray]:
    """
    Apply the translation (if any) of the given SpatialImage to its x- and y-coordinates
    array. The new SpatialImage is returned, as well as the original coordinates array.
    This function is used because some plotting functions ignore the SpatialImage transformation
    matrix, but do use the coordinates arrays for absolute positioning of the image in the plot.
    After plotting the coordinates can be restored with _unapply_transform().
    """
    # Get current coords
    x_orig_coords = se.x.data
    y_orig_coords = se.y.data

    # Translate
    tx, ty = _get_translation(se)
    x_coords = xr.DataArray(tx + np.arange(se.sizes["x"], dtype="float64"), dims="x")
    y_coords = xr.DataArray(ty + np.arange(se.sizes["y"], dtype="float64"), dims="y")
    se = se.assign_coords({"x": x_coords, "y": y_coords})
    # QUESTION: should we set the resulting SpatialImage's transformation matrix to the
    # identity matrix too, for consistency? If so we have to keep track of it too for restoring later.

    return se, x_orig_coords, y_orig_coords


def _unapply_transform(
    se: Union[SpatialImage, DataArray], x_coords: np.ndarray, y_coords: np.ndarray
) -> Union[SpatialImage, DataArray]:
    """
    Restore the coordinates which were temporarily modified via _apply_transform().
    """
    se = se.assign_coords({"y": y_coords, "x": x_coords})
    return se


def _get_spatial_element(
    sdata: SpatialData, layer: str
) -> Union[SpatialImage, DataArray]:
    if layer in sdata.images:
        si = sdata.images[layer]
    elif layer in sdata.labels:
        si = sdata.labels[layer]
    else:
        raise KeyError(f"'{layer}' not found in sdata.images or sdata.labels")
    if isinstance(si, SpatialImage):
        return si
    elif isinstance(si, MultiscaleSpatialImage):
        # get the name of the unscaled image
        # TODO maybe add some other checks here
        scale_0 = si.__iter__().__next__()
        name=si[scale_0].__iter__().__next__()
        return si[scale_0][name]
    else:
        raise ValueError(f"Not implemented for layer '{layer}' of type {type(si)}.")


def _get_transformation(
    element: Union[SpatialElement, DataArray],
    to_coordinate_system: Optional[str] = None,
    get_all: bool = False,
) -> Union[BaseTransformation, dict[str, BaseTransformation]]:
    """
    Get the transformation/s of an element.

    This function extends the capabilities of `spatialdata.transformations.get_transformation` by also supporting extraction from `xarray.DataArray`.
    This facilitates interaction with `MultiscaleSpatialImage`.

    Parameters
    ----------
    element
        The element.
    to_coordinate_system
        The coordinate system to which the transformation should be returned.

        * If None and `get_all=False` returns the transformation from the 'global' coordinate system (default system).
        * If None and `get_all=True` returns all transformations.

    get_all
        If True, all transformations are returned. If True, `to_coordinate_system` needs to be None.

    Returns
    -------
    The transformation, if `to_coordinate_system` is not None, otherwise a dictionary of transformations to all
    the coordinate systems.
    """
    from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM

    if isinstance(element, SpatialElement):
        transformations = _get_transformations(element)
    elif isinstance(element, DataArray):
        transformations = _get_transformations_xarray(element)
    assert isinstance(transformations, dict)

    if get_all is False:
        if to_coordinate_system is None:
            to_coordinate_system = DEFAULT_COORDINATE_SYSTEM
        # get a specific transformation
        if to_coordinate_system not in transformations:
            raise ValueError(
                f"Transformation to {to_coordinate_system} not found in element {element}."
            )
        return transformations[to_coordinate_system]
    else:
        assert to_coordinate_system is None
        # get the dict of all the transformations
        return transformations


def _add_image_layer(
    sdata: SpatialData,
    arr: Array,
    output_layer: str,
    chunks: Optional[str | tuple[int, int, int] | int] = None,
    transformation: Union[BaseTransformation, dict[str, BaseTransformation]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
):
    manager = ImageLayerManager()
    sdata = manager.add_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=chunks,
        transformation=transformation,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata

def _add_label_layer(
    sdata: SpatialData,
    arr: Array,
    output_layer: str,
    chunks: Optional[str | tuple[int, int] | int] = None,
    transformation: Union[BaseTransformation, dict[str, BaseTransformation]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
):
    manager = LabelLayerManager()
    sdata = manager.add_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=chunks,
        transformation=transformation,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
