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
                f"The overlapping depth '{_depth}' is larger than your array '{min_size}'. Setting depth to 'chunksize_clahe//4 ({depth}')"
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
