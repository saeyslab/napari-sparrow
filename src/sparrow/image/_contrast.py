from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image._apply import apply
from sparrow.image._image import _get_spatial_element


def enhance_contrast(
    sdata: SpatialData,
    img_layer: str | None = None,
    contrast_clip: float | list[float] = 3.5,
    chunks: str | tuple[int, ...] | int | None = 10000,
    depth: tuple[int, ...] | dict[int, int] | int = 3000,
    output_layer: str = "clahe",
    crd: tuple[int, int, int, int] | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Enhance the contrast of an image in a SpatialData object using
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Compatibility with image layers that have either two or three spatial dimensions (c, (z), y, x).

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the image to enhance.
    img_layer : Optional[str], default=None
        The image layer in `sdata` on which the enhance_contrast function will be applied.
        If not provided, the last image layer in `sdata` is used.
    contrast_clip : Union[float, List[float]], optional
        The clip limit for the CLAHE algorithm. Higher values result in stronger contrast enhancement
        but also stronger noise amplification.
        If provided as a list, the length must match the number of channels,
        as the parameter will be used to process the different channels.
        The default value is 3.5.
    chunks : str | Tuple[int,...] | int, optional
        The size of the chunks used during dask image processing.
        The default value is 10000.
    depth : Tuple[int, ...] | Dict[ int, int ] | int, optional
        The overlapping depth used in dask array map_overlap operation.
        The default value is 3000.
    output_layer : str, optional
        The name of the image layer where the enhanced image will be stored.
        The default value is "clahe".
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite: bool
        If True overwrites the element if it already exists.

    Returns
    -------
    SpatialData
        An updated `sdata` object with the contrast enhanced image added as a new layer.

    Raises
    ------
    ValueError
        If the dimensions in `img_layer` of `sdata` is not equal to (c,(z),y,x)

    Notes
    -----
    CLAHE is applied to each channel and z-stack (if the image is 3D) of the image separately.
    """

    def _apply_clahe(image: NDArray, contrast_clip: float = 3.5) -> NDArray:
        # input c, (z) , y, x
        # output c, (z), y, x
        # squeeze dimension, and then put it back

        image_dim = image.ndim
        if image_dim == 3:
            if image.shape[0] == 1:
                image = np.squeeze(image, axis=0)
            else:
                raise ValueError("_apply_clahe only accepts c dimension equal to 1.")
        elif image_dim == 4:
            if image.shape[0] == 1 and image.shape[1] == 1:
                image = np.squeeze(image, axis=(0, 1))
            else:
                raise ValueError("_apply_clahe only accepts c and z dimension equal to 1.")
        else:
            raise ValueError("Please provide numpy array containing c,(z),y and x dimension.")

        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        image = clahe.apply(image)

        if image_dim == 3:
            image = image[None, ...]
        else:
            image = image[None, None, ...]

        return image

    se = _get_spatial_element(sdata, img_layer)

    if isinstance(contrast_clip, Iterable):
        assert (
            len(contrast_clip) == len(se.c.data)
        ), f"If 'contrast_clip' is provided as a list, it should match the number of channels in '{se}' ({len(se.c.data)})"
        fn_kwargs = {key: {"contrast_clip": value} for (key, value) in zip(se.c.data, contrast_clip)}
    else:
        fn_kwargs = {"contrast_clip": contrast_clip}

    if se.dims == ("c", "z", "y", "x"):
        if isinstance(depth, int):
            depth = (0, 0, depth, depth)
    elif se.dims == ("c", "y", "x"):
        if isinstance(depth, int):
            depth = (0, depth, depth)
    else:
        raise ValueError(f"Dimensions for provided img_layer are {se.dims}. We only support (c, (z), y, x).")

    sdata = apply(
        sdata,
        _apply_clahe,
        fn_kwargs=fn_kwargs,
        img_layer=img_layer,
        output_layer=output_layer,
        combine_c=False,  # you want to process all channels independently.
        combine_z=False,  # you want to process all z-stacks independently.
        chunks=chunks,  # provide it for c,z,y,x,
        depth=depth,  # provide it for c,z,y,x; depth > 0 for c and z will be ignored
        crd=crd,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
