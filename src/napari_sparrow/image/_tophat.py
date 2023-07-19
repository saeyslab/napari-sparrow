import dask_image
from spatialdata import SpatialData
import dask.array as da
import spatialdata
from spatialdata.transformations import set_transformation, get_transformation


# TODO: rename this file and rename tophat_filtering


def tophat_filtering(
    sdata: SpatialData,
    output_layer="tophat_filtered",
    size_tophat: int = 85,
) -> SpatialData:
    # this is function to do tophat filtering using dask

    # take the last image as layer to do next step in pipeline
    layer = [*sdata.images][-1]

    # TODO size_tophat maybe different for different channels, probably fix this with size_tophat as a list.
    # Initialize list to store results
    result_list = []

    for channel in sdata[layer].c.data:
        image_array = sdata[layer].isel(c=channel).data

        # Apply the minimum filter
        minimum_t = dask_image.ndfilters.minimum_filter(image_array, size_tophat)

        # Apply the maximum filter
        max_of_min_t = dask_image.ndfilters.maximum_filter(minimum_t, size_tophat)

        # Subtract max_of_min_t from image_array and store in result_list
        result_list.append(image_array - max_of_min_t)

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))
    trf = get_transformation(sdata[layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
