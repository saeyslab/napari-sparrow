from spatialdata import SpatialData
import dask.array as da
import spatialdata
import squidpy as sq
import cv2
import warnings
from spatialdata.transformations import set_transformation, get_transformation


def clahe_processing(
    sdata: SpatialData,
    output_layer: str = "clahe",
    contrast_clip: int = 3.5,   # FIXME: is contrast_clip an integer or a float?
    chunksize_clahe: int = 10000,
    depth: int = 3000,
) -> SpatialData:
    # TODO take whole image as chunksize + overlap tuning

    layer = [*sdata.images][-1]

    # set depth
    min_size = min(sdata[layer].sizes["x"], sdata[layer].sizes["y"])
    _depth = depth
    if min_size < depth:
        if min_size < chunksize_clahe // 4:
            depth = min_size // 4
            warnings.warn(
                f"The overlapping depth '{_depth}' is larger than your array '{min_size}'. Setting depth to 'min_size//4 ({depth}')"
            )

        else:
            depth = chunksize_clahe // 4
            warnings.warn(
                f"The overlapping depth '{_depth}' is larger than your array '{min_size}'. Setting depth to 'chunksize_clahe//4 ({depth}')"
            )

    # convert to imagecontainer, because apply not yet implemented in sdata
    ic = sq.im.ImageContainer(sdata[layer], layer=layer)

    result_list = []

    for channel in sdata[layer].c.data:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))

        ic_clahe = ic.apply(
            {"0": clahe.apply},
            layer=layer,
            new_layer=output_layer,
            drop=True,
            channel=channel,
            copy=True,
            chunks=chunksize_clahe,
            lazy=True,
            depth=depth,
            boundary="reflect",
        )

        # squeeze channel dim and z-dimension
        result_list.append(ic_clahe["clahe"].data.squeeze(axis=(2, 3)))

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))
    trf = get_transformation(sdata[layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
