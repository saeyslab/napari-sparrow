import warnings

import cv2
import dask.array as da
import spatialdata
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation, set_transformation


def enhance_contrast(
    sdata: SpatialData,
    output_layer: str = "clahe",
    contrast_clip: float = 3.5,
    chunks: int = 10000,
    depth: int = 3000,
) -> SpatialData:
    """
    Enhance the contrast of an image in a SpatialData object using 
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the image to enhance.
    output_layer : str, optional
        The name of the layer where the enhanced image will be stored.
        The default value is "clahe".
    contrast_clip : float, optional
        The clip limit for the CLAHE algorithm. Higher values result in stronger contrast enhancement
        but also stronger noise amplification.
        The default value is 3.5.
    chunks : int, optional
        The size of the chunks used during dask image processing.
        Larger chunks may lead to increased memory usage but faster processing.
        The default value is 10000.
    depth : int, optional
        The overlapping depth used in dask array map_overlap operation.
        The default value is 3000.

    Returns
    -------
    SpatialData
        A new SpatialData object with the contrast enhanced image added as a new layer.

    Notes
    -----
    CLAHE is applied to each channel of the image separately.
    """    

    layer = [*sdata.images][-1]

    # set depth
    min_size = min(sdata[layer].sizes["x"], sdata[layer].sizes["y"])
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

    def _apply_clahe(image):
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
        return clahe.apply(image)

    result_list = []

    for channel in sdata[layer].c.data:
        arr = sdata[layer].isel(c=channel).data
        arr = arr.rechunk(chunks)
        result = arr.map_overlap(
            _apply_clahe, dtype=sdata[layer].data.dtype, depth=depth, boundary="reflect"
        )
        result = result.rechunk(chunks)
        result_list.append(result)

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))
    trf = get_transformation(sdata[layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
