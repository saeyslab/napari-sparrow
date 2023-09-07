import warnings
from typing import List, Optional

import cv2
import dask.array as da
import spatialdata
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation, set_transformation


def enhance_contrast(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    contrast_clip: float | List[float] = 3.5,
    chunks: int = 10000,
    depth: int = 3000,
    output_layer: str = "clahe",
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
    chunks : int, optional
        The size of the chunks used during dask image processing.
        Larger chunks may lead to increased memory usage but faster processing.
        The default value is 10000.
    depth : int, optional
        The overlapping depth used in dask array map_overlap operation.
        The default value is 3000.
    output_layer : str, optional
        The name of the image layer where the enhanced image will be stored.
        The default value is "clahe".

    Returns
    -------
    SpatialData
        An updated `sdata` object with the contrast enhanced image added as a new layer.

    Notes
    -----
    CLAHE is applied to each channel of the image separately.
    """

    if img_layer is None:
        img_layer = [*sdata.images][-1]
    img_layer = [*sdata.images][-1]

    # Check if contrast_clip is a list and if its size is the same as the number of channels
    if isinstance(contrast_clip, list) and len(contrast_clip) != len(
        sdata[img_layer].c.data
    ):
        raise ValueError(
            "Size of contrast_clip list must be the same as the number of channels."
        )

    # set depth
    min_size = min(sdata[img_layer].sizes["x"], sdata[img_layer].sizes["y"])
    _depth = depth
    if min_size < depth:
        if min_size < chunks // 4:
            depth = min_size // 4
            warnings.warn(
                f"The overlapping depth '{_depth}' is larger than your array '{min_size}'. Setting depth to 'min_size//4 ({depth}')"
            )

        else:
            depth = chunks // 4
            warnings.warn(
                f"The overlapping depth '{_depth}' is larger than your array '{min_size}'. Setting depth to 'chunks//4 ({depth}')"
            )

    def _apply_clahe(image, contrast_clip):
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        return clahe.apply(image)

    result_list = []

    for channel_idx, channel in enumerate(sdata[img_layer].c.data):
        arr = sdata[img_layer].isel(c=channel).data
        arr = arr.rechunk(chunks)
        current_contrast_clip = (
            contrast_clip[channel_idx]
            if isinstance(contrast_clip, list)
            else contrast_clip
        )
        result = arr.map_overlap(
            _apply_clahe,
            dtype=sdata[img_layer].data.dtype,
            depth=depth,
            boundary="reflect",
            contrast_clip=current_contrast_clip,
        )
        result = result.rechunk(chunks)
        result_list.append(result)

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))
    trf = get_transformation(sdata[img_layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
