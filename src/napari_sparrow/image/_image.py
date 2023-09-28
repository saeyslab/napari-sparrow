import warnings
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from spatial_image import SpatialImage
from spatialdata.transformations import get_transformation


def _substract_translation_crd(
    spatial_image: SpatialImage, crd=Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
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


def _get_boundary(spatial_image: SpatialImage) -> Tuple[int, int, int, int]:
    tx, ty = _get_translation(spatial_image)
    width = spatial_image.sizes["x"]
    height = spatial_image.sizes["y"]
    return (int(tx), int(tx + width), int(ty), int(ty + height))


def _get_translation(spatial_image: SpatialImage) -> Tuple[float, float]:
    transform_matrix = get_transformation(spatial_image).to_affine_matrix(
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


# FIXME: the type "SpatialImage" is probably too restrictive here, see type of sdata[layer], which is more general
def _apply_transform(si: SpatialImage) -> Tuple[SpatialImage, np.ndarray, np.ndarray]:
    """
    Apply the translation (if any) of the given SpatialImage to its x- and y-coordinates
    array. The new SpatialImage is returned, as well as the original coordinates array.
    This function is used because some plotting functions ignore the SpatialImage transformation
    matrix, but do use the coordinates arrays for absolute positioning of the image in the plot.
    After plotting the coordinates can be restored with _unapply_transform().
    """
    # Get current coords
    x_orig_coords = si.x.data
    y_orig_coords = si.y.data

    # Translate
    tx, ty = _get_translation(si)
    x_coords = xr.DataArray(tx + np.arange(si.sizes["x"], dtype="float64"), dims="x")
    y_coords = xr.DataArray(ty + np.arange(si.sizes["y"], dtype="float64"), dims="y")
    si = si.assign_coords({"x": x_coords, "y": y_coords})
    # QUESTION: should we set the resulting SpatialImage's transformation matrix to the
    # identity matrix too, for consistency? If so we have to keep track of it too for restoring later.

    return si, x_orig_coords, y_orig_coords


def _unapply_transform(
    si: SpatialImage, x_coords: np.ndarray, y_coords: np.ndarray
) -> SpatialImage:
    """
    Restore the coordinates which were temporarily modified via _apply_transform().
    """
    si = si.assign_coords({"y": y_coords, "x": x_coords})
    return si
