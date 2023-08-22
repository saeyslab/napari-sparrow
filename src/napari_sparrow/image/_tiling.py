from typing import List, Optional, Tuple

import cv2
import dask.array as da
import numpy as np
import spatialdata
import squidpy as sq
from basicpy import BaSiC
from spatialdata import SpatialData
from spatialdata.transformations import Translation, set_transformation

from napari_sparrow.image._image import _get_translation, _substract_translation_crd
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def tiling_correction(
    sdata: SpatialData,
    tile_size: int = 2144,
    crd: Optional[Tuple[int, int, int, int]] = None,
    output_layer: str = "tiling_correction",
) -> Tuple[SpatialData, List[np.ndarray]]:
    """Returns the corrected image and the flatfield array

    This function corrects for the tiling effect that occurs in some image data for example the resolve dataset.
    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.
    """

    layer = [*sdata.images][-1]

    if sdata[layer].sizes["x"] % tile_size or sdata[layer].sizes["y"] % tile_size:
        raise ValueError(
            f"Dimension of image layer '{layer}' ({sdata[layer].shape}) on which to run the "
            f"tilingCorrection is not a multiple of the given tile size ({tile_size})."
        )

    # crd is specified on original uncropped pixel coordinates
    # need to substract possible translation, because we use crd to crop imagecontainer, which does not take
    # translation into account
    if crd:
        crd = _substract_translation_crd(sdata[layer], crd)

    tx, ty = _get_translation(sdata[layer])

    result_list = []
    flatfields = []

    for channel in sdata[layer].c.data:
        ic = sq.im.ImageContainer(sdata[layer].isel(c=channel), layer=layer)

        # Create the tiles
        tiles = ic.generate_equal_crops(size=tile_size, as_array=layer)
        tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])
        black = np.array(
            [1 if ~np.any(tile - 1) else 0 for tile in tiles]
        )  # determine if

        # create the masks for inpainting
        i_mask = (
            np.block(
                [
                    list(tiles[i : i + (ic.shape[1] // tile_size)])
                    for i in range(0, len(tiles), ic.shape[1] // tile_size)
                ]
            ).astype(np.uint16)
            == 0
        )
        if tiles.shape[0] < 5:
            log.info(
                "There aren't enough tiles to perform tiling correction (less than 5). This step will be skipped."
            )
            tiles_corrected = tiles
            flatfields.append(None)
        else:
            basic = BaSiC(smoothness_flatfield=1)
            basic.fit(tiles)
            flatfields.append(basic.flatfield)
            tiles_corrected = basic.transform(tiles)

        tiles_corrected = np.array(
            [
                tiles[number] if black[number] == 1 else tile
                for number, tile in enumerate(tiles_corrected)
            ]
        )

        # Stitch the tiles back together
        i_new = np.block(
            [
                list(tiles_corrected[i : i + (ic.shape[1] // tile_size)])
                for i in range(0, len(tiles_corrected), ic.shape[1] // tile_size)
            ]
        ).astype(np.uint16)

        ic = sq.im.ImageContainer(i_new, layer=layer)
        ic.add_img(
            i_mask.astype(np.uint8),
            layer="mask_black_lines",
        )

        if crd:
            x0 = crd[0]
            x_size = crd[1] - crd[0]
            y0 = crd[2]
            y_size = crd[3] - crd[2]
            ic = ic.crop_corner(y=y0, x=x0, size=(y_size, x_size))

        # Perform inpainting
        ic.apply(
            {"0": cv2.inpaint},
            layer=layer,
            drop=False,
            channel=0,
            new_layer=output_layer,
            copy=False,
            # chunks=10,
            fn_kwargs={
                "inpaintMask": ic.data.mask_black_lines.squeeze().to_numpy(),
                "inpaintRadius": 55,
                "flags": cv2.INPAINT_NS,
            },
        )

        # result for each channel
        result_list.append(ic[output_layer].data)

    # make one dask array of shape (c,y,x)
    result = da.concatenate(result_list, axis=-1).transpose(3, 0, 1, 2).squeeze(-1)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))

    if crd:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))
    set_transformation(spatial_image, translation)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata, flatfields
