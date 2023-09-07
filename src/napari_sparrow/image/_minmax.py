from typing import List, Optional

import dask.array as da
import dask_image
import spatialdata
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation, set_transformation


def min_max_filtering(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    size_min_max_filter: int | List[int] = 85,
    output_layer="min_max_filtered",
) -> SpatialData:
    """
    Apply min max filtering to the given SpatialData object using dask.

    The function accepts a SpatialData object and applies min max filtering
    to the last image layer in the object. The size of the filter can be provided
    either as an integer or a list of integers corresponding to each channel.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the images to be processed.
    img_layer : Optional[str], default=None
        The image layer in `sdata` to run min_max_filtering on. If not provided, the last image layer in `sdata` is used.
    size_min_max_filter : Union[int, List[int]], optional
        Size of the min_max filter. If provided as a list, the length
        must match the number of channels. Defaults to 85.
    output_layer : str, optional
        The name of the output layer. Defaults to "min_max_filtered".

    Returns
    -------
    SpatialData
        The `sdata` object with the min_max-filtered image added.

    Raises
    ------
    ValueError
        If `size_min_max_filter` is a list and its length does not match the number of channels.

    Examples
    --------
    Apply min max filtering with a single size:

    >>> sdata = min_max_filtering(sdata, size_min_max_filtering=50)

    Apply min max filtering with different sizes for each channel:

    >>> sdata = min_max_filtering(sdata, size_min_max_filtering=[30, 50, 70])
    """
    # take the last image as layer to do next step in pipeline
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    # Check if size_min_max_filter is a list and if its size is the same as the number of channels
    if isinstance(size_min_max_filter, list) and len(size_min_max_filter) != len(
        sdata[img_layer].c.data
    ):
        raise ValueError(
            "Size of size_min_max_filter list must be the same as the number of channels."
        )

    # Initialize list to store results
    result_list = []

    for channel_idx, channel in enumerate(sdata[img_layer].c.data):
        image_array = sdata[img_layer].isel(c=channel).data

        # Determine the size_min_max_filter for the current channel
        current_size_min_max_filter = (
            size_min_max_filter[channel_idx]
            if isinstance(size_min_max_filter, list)
            else size_min_max_filter
        )

        # Apply the minimum filter
        minimum_t = dask_image.ndfilters.minimum_filter(
            image_array, current_size_min_max_filter
        )

        # Apply the maximum filter
        max_of_min_t = dask_image.ndfilters.maximum_filter(
            minimum_t, current_size_min_max_filter
        )

        # Subtract max_of_min_t from image_array and store in result_list
        result_list.append(image_array - max_of_min_t)

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))
    trf = get_transformation(sdata[img_layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    return sdata
