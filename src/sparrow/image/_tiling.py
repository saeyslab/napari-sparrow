from __future__ import annotations

import dask.array as da
import numpy as np
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation, get_transformation

from sparrow.image._image import (
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
    add_image_layer,
)
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    from basicpy import BaSiC
except ImportError:
    # Assign None to BaSiC to avoid NameError in the function signature, but ignore type checking for this assignment.
    # Without assigning None, the function could later fail with an unclear NameError when it tried to use BaSiC
    BaSiC = None  # type: ignore[assignment,misc]
    log.warning(
        "'basicpy' not installed, to use 'sparrow.im.tiling_correction', please install this library."
    )

try:
    import cv2
except ImportError:
    # Assign None to cv2 to avoid NameError in the function signature, but ignore type checking for this assignment.
    cv2 = None  # type: ignore[assignment]
    log.warning("'OpenCV (cv2)' not installed, to use 'sparrow.im.tiling_correction' please install this library.")


def _materialize_array(array: np.ndarray | da.Array) -> np.ndarray:
    """Convert a possibly lazy image array to a NumPy array."""
    # Evaluate a Dask-backed image slice before passing it to NumPy-only dependencies.
    if isinstance(array, da.Array):
        array = array.compute()

    # Normalize the resulting image slice to a plain NumPy array.
    return np.asarray(array)


def _stitch_tiles(tiles: np.ndarray, tile_rows: int, tile_columns: int) -> np.ndarray:
    """Reassemble row-major square tiles into a two-dimensional image."""
    # Group tiles by their row and column positions before moving pixel axes next to each other.
    tiled = tiles.reshape(tile_rows, tile_columns, tiles.shape[1], tiles.shape[2])

    # Flatten the interleaved tile grid into the original image height and width.
    return tiled.transpose(0, 2, 1, 3).reshape(
        tile_rows * tiles.shape[1],
        tile_columns * tiles.shape[2],
    )


def tiling_correction(
    sdata: SpatialData,
    img_layer: str | None = None,
    tile_size: int = 2144,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    scale_factors: ScaleFactors_t | None = None,
    output_layer: str = "tiling_correction",
    overwrite: bool = False,
) -> tuple[SpatialData, list[np.ndarray]]:
    """
    Function corrects for the tiling effect that occurs in some image data (e.g. resolve data).

    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.

    Parameters
    ----------
    sdata
        The SpatialData object containing the image data to correct.
    img_layer
        The image layer in `sdata` to be corrected for tiling effects. If not provided, the last image layer in `sdata` is used.
    tile_size
        The size of the tiles in the image.
    crd
        Coordinates defining the region of the image to correct. It defines the bounds (x_min, x_max, y_min, y_max).
    to_coordinate_system
        The coordinate system to which the `crd` is specified. Ignored if `crd` is None.
        If a `crd` is specified, only the coordinate system defined here will be kept in `output_layer`.
    scale_factors
        Scale factors to apply for multiscale.
    output_layer
        Name of the image layer where the corrected image will be stored in the `sdata` object.
    overwrite
        If True overwrites the element if it already exists.

    Returns
    -------
    Updated `sdata` object containing the corrected image and a list of flatfield arrays with length equal to the number of channels.

    Raises
    ------
    ValueError
        If the image layer does not contain exactly 2 spatial dimensions.
    ValueError
        If the dimensions of the image layer are not multiples of the given tile size.

    Notes
    -----
    The function integrates the BaSiC algorithm for illumination correction and uses OpenCV's inpainting
    to stitch tiles together. It manages the pre- and post-processing of data, translation of coordinates,
    and addition of corrected image results back to the `sdata` object.
    """
    # Guard against missing optional dependencies before any image processing starts.
    if BaSiC is None or cv2 is None:
        raise ImportError(
            "'basicpy' and 'opencv-python' are required for tiling_correction. "
            "Install them with: `uv sync --extra tiling`"
        )

    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Applying image processing on the last image layer '{img_layer}' of the provided SpatialData object."
        )

    se = _get_spatial_element(sdata, layer=img_layer)

    if se.dims != ("c", "y", "x"):
        raise ValueError(
            "Tiling correction is only supported for images with 2 spatial dimensions, "
            f"while provided image layer ({img_layer}) has dimensions {se.ndim}."
        )

    if se.sizes["x"] % tile_size or se.sizes["y"] % tile_size:
        raise ValueError(
            f"Spatial Dimension of image layer '{img_layer}' ({se.shape}) on which to run the "
            f"tilingCorrection is not a multiple of the given tile size ({tile_size})."
        )

    # crd is specified on original uncropped pixel coordinates
    # need to substract possible translation, because we use crd to crop imagecontainer, which does not take
    # translation into account
    if crd is not None:
        crd = _substract_translation_crd(spatial_image=se, crd=crd, to_coordinate_system=to_coordinate_system)
        tx, ty = _get_translation(se, to_coordinate_system=to_coordinate_system)

    # Calculate the number of complete tiles along each spatial dimension.
    tile_rows = se.sizes["y"] // tile_size
    tile_columns = se.sizes["x"] // tile_size

    # Keep corrected channel results and BaSiC flatfields in channel order.
    result_list = []
    flatfields = []

    for channel_idx, channel in enumerate(se.c.data):
        # Materialize one channel so BaSiC can fit the illumination model on its tiles.
        channel_data = _materialize_array(se.isel(c=channel_idx).data)

        # Extract tiles in the same row-major order used by Squidpy's image container.
        # The intermediate reshape has this conceptual layout: (row, pixel_y, column, pixel_x)
        tiles = channel_data.reshape(tile_rows, tile_size, tile_columns, tile_size)
        # The transpose changes it to: (row, column, pixel_y, pixel_x) so that the final reshape flattens the first two axes into a single tile index.
        tiles = tiles.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)

        # Shift completely black tiles so BaSiC can process them without changing their output later.
        # The reason for the shift is that BaSiC should not interpret a completely black background tile as an illumination pattern.
        tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])
        # After shifting an all-zero tile by one, the code identifies it with a boolean mask so it can be restored to its original state after BaSiC processing.
        black = np.all(tiles == 1, axis=(1, 2))

        # Mark zero-valued pixels inside non-black tiles for inter-tile inpainting.
        i_mask = _stitch_tiles(tiles == 0, tile_rows, tile_columns).astype(np.uint8)

        # Fit BaSiC to estimate and remove per-tile illumination variation.
        basic = BaSiC(smoothness_flatfield=1)
        basic.fit(tiles)
        if np.isnan(basic._reweight_score).item():
            log.warning(
                f"Basicpy model used for illumination correction for channel '{channel}' did not converge. "
                "Illumination correction will be skipped. Continuing with inpainting. Please consider using a larger image ( more tiles )."
            )
            flatfields.append(None)
            # Making sure inpainting can still be performed even when illumination correction is unavailable for a particular channel
            tiles_corrected = tiles
        else:
            flatfields.append(basic.flatfield)
            tiles_corrected = basic.transform(tiles)

        # Restore completely black tiles because they are background rather than illumination samples.
        tiles_corrected = np.asarray(tiles_corrected)
        tiles_corrected[black] = tiles[black]

        # Stitch the corrected tiles back into a full two-dimensional channel image.
        i_new = _stitch_tiles(tiles_corrected, tile_rows, tile_columns).astype(np.uint16)

        # Crop both the corrected image and its inpainting mask when a region was requested.
        if crd is not None:
            x0 = crd[0]
            x_size = crd[1] - crd[0]
            y0 = crd[2]
            y_size = crd[3] - crd[2]
            i_new = i_new[y0 : y0 + y_size, x0 : x0 + x_size]
            i_mask = i_mask[y0 : y0 + y_size, x0 : x0 + x_size]

        # Fill masked inter-tile lines using the same Navier-Stokes inpainting algorithm.
        corrected_image = cv2.inpaint(
            i_new,
            i_mask,
            55,
            cv2.INPAINT_NS,
        )

        # Store the corrected channel for Dask reassembly after all channels are processed.
        result_list.append(da.from_array(corrected_image))

    # Make one Dask array with the SpatialData channel-y-x dimension order.
    result = da.stack(result_list, axis=0)

    if crd is not None:
        tx = tx + crd[0]
        ty = ty + crd[2]

        # Create a translation transformation to account for the cropping and any existing translation.
        translation = Translation([tx, ty], axes=("x", "y"))

    else:
        translation = None
        transformations = get_transformation(se, get_all=True)

    sdata = add_image_layer(
        sdata,
        arr=result,
        output_layer=output_layer,
        chunks=result.chunksize,
        transformations={to_coordinate_system: translation} if translation is not None else transformations,
        scale_factors=scale_factors,
        c_coords=se.c.data,
        overwrite=overwrite,
    )

    return sdata, flatfields
