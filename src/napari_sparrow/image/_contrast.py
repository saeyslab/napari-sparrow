from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from numpy.typing import NDArray
import numpy as np
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from napari_sparrow.image._apply import ChannelList, apply
from napari_sparrow.image._image import _get_spatial_element


def enhance_contrast(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    contrast_clip: float | List[float] = 3.5,
    chunks: Optional[str | Tuple[int, ...] | int] = 10000,
    depth: Tuple[int, ...] | Dict[int, int] | int = 3000,
    output_layer: str = "clahe",
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Enhance the contrast of an image in a SpatialData object using
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Compatibility with image layers that have either two or three spatial dimensions.

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
        If provided as a list, the length must match the number of channels
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
    -------
    ValueError
        - If the number of dimensions in `img_layer` of `sdata` > 4
        - If Depth not equal to 0 for 'z' dimension, if there are 3 spatial dimension.
        - If the number of spatial dimensions is not equal to 2 or 2.

    Notes
    -----
    CLAHE is applied to each channel of the image separately.
    For 3D images (3 spatial dimensions), CLAHE is applied to each z-slice separately.
    For large 3D images (3 spatial dimensions), we advice to set the chunks parameter to (1, ..., ...).
    """

    def _apply_clahe(image: NDArray, contrast_clip: float = 3.5) -> NDArray:
        if image.ndim == 3:
            processed_slices = [
                _apply_clahe_2d(image_slice, contrast_clip=contrast_clip)
                for image_slice in image
            ]
            processed_image = np.stack(processed_slices, axis=0)
            return processed_image
        elif image.ndim == 2:
            return _apply_clahe_2d(image, contrast_clip)
        else:
            raise ValueError(
                f"Only 2D ad 3D arrays are supported, but provided array was of dimension '{image.ndim}'"
            )

    def _apply_clahe_2d(image: NDArray, contrast_clip: float = 3.5) -> NDArray:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        return clahe.apply(image)

    if isinstance(contrast_clip, Iterable) and not isinstance(contrast_clip, str):
        contrast_clip = ChannelList(contrast_clip)

    se = _get_spatial_element(sdata, img_layer)

    if len(se.dims) > 4:
        raise ValueError(
            f"Dimensions for provided img_layer are {se.dims}. We only support (c, (z), y, x)."
        )

    elif len(se.dims) == 4:
        if isinstance(depth, int):
            # we do not allow depth!=0 for z dimension for enhance contrast
            depth = (0, depth, depth)
            # if depth is not an int, coerce depth will fix it.
        assert depth[0] == 0, "Depth not equal to 0 for 'z' dimension is not supported."

    sdata = apply(
        sdata,
        _apply_clahe,
        img_layer=img_layer,
        output_layer=output_layer,
        chunks=chunks,
        channel=None,  # channel==None -> apply apply_clahe to each layer seperately
        fn_kwargs={"contrast_clip": contrast_clip},
        depth=depth,
        crd=crd,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
