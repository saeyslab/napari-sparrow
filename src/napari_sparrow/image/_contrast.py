from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from numpy.typing import NDArray
from spatialdata import SpatialData

from napari_sparrow.image._apply import ChannelList, apply


def enhance_contrast(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    contrast_clip: float | List[float] = 3.5,
    chunks: Optional[str | tuple[int, int] | int] = 10000,
    depth: Tuple[int, int] | Dict[int, int] | int = 3000,
    output_layer: str = "clahe",
    crd: Optional[Tuple[int, int, int, int]] = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Enhance the contrast of an image in a SpatialData object using
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

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
    chunks : str | tuple[int, int] | int, optional
        The size of the chunks used during dask image processing.
        The default value is 10000.
    depth : Tuple[int, int] | Dict[ int, int ] | int, optional
        The overlapping depth used in dask array map_overlap operation.
        The default value is 3000.
    output_layer : str, optional
        The name of the image layer where the enhanced image will be stored.
        The default value is "clahe".
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    overwrite: bool
        If True overwrites the element if it already exists.

    Returns
    -------
    SpatialData
        An updated `sdata` object with the contrast enhanced image added as a new layer.

    Notes
    -----
    CLAHE is applied to each channel of the image separately.
    """

    def _apply_clahe(image: NDArray, contrast_clip: float = 3.5) -> NDArray:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        return clahe.apply(image)

    if isinstance(contrast_clip, Iterable) and not isinstance(contrast_clip, str):
        contrast_clip = ChannelList(contrast_clip)

    sdata = apply(
        sdata,
        _apply_clahe,
        img_layer=img_layer,
        output_layer=output_layer,
        chunks=chunks,
        channel=None, # channel==None -> apply apply_clahe to each layer seperately
        fn_kwargs={"contrast_clip": contrast_clip},
        depth=depth,
        crd=crd,
        overwrite=overwrite,
    )

    return sdata
