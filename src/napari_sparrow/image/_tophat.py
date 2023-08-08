from typing import List

import dask.array as da
import dask_image
import spatialdata
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation, set_transformation

# TODO: rename this file and rename tophat_filtering


def tophat_filtering(
    sdata: SpatialData,
    output_layer="tophat_filtered",
    size_tophat: int | List[int] = 85,
) -> SpatialData:
    """
    Apply tophat filtering to the given SpatialData object using dask.

    The function accepts a SpatialData object and applies tophat filtering
    to the last image layer in the object. The size of the tophat can be provided
    either as an integer or a list of integers corresponding to each channel.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the images to be processed.
    output_layer : str, optional
        The name of the output layer. Defaults to "tophat_filtered".
    size_tophat : Union[int, List[int]], optional
        Size of the tophat for filtering. If provided as a list, the length
        must match the number of channels. Defaults to 85.

    Returns
    -------
    SpatialData
        The SpatialData object with the tophat-filtered image added.

    Raises
    ------
    ValueError
        If `size_tophat` is a list and its length does not match the number of channels.

    Examples
    --------
    Apply tophat filtering with a single size:

    >>> sdata = tophat_filtering(sdata, size_tophat=50)

    Apply tophat filtering with different sizes for each channel:

    >>> sdata = tophat_filtering(sdata, size_tophat=[30, 50, 70])
    """

    # this is function to do tophat filtering using dask

    # take the last image as layer to do next step in pipeline
    layer = [*sdata.images][-1]

    # Check if size_tophat is a list and if its size is the same as the number of channels
    if isinstance(size_tophat, list) and len(size_tophat) != len(sdata[layer].c.data):
        raise ValueError("Size of size_tophat list must be the same as the number of channels.")

    # Initialize list to store results
    result_list = []

    for channel_idx, channel in enumerate( sdata[layer].c.data):
        image_array = sdata[layer].isel(c=channel).data

        # Determine the size_tophat for the current channel
        current_size_tophat = size_tophat[channel_idx] if isinstance(size_tophat, list) else size_tophat

        # Apply the minimum filter
        minimum_t = dask_image.ndfilters.minimum_filter(image_array, current_size_tophat )

        # Apply the maximum filter
        max_of_min_t = dask_image.ndfilters.maximum_filter(minimum_t, current_size_tophat )

        # Subtract max_of_min_t from image_array and store in result_list
        result_list.append(image_array - max_of_min_t)

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))
    trf = get_transformation(sdata[layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
