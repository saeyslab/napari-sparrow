from typing import Iterable, List, Optional

from dask.array import Array
from dask_image.ndfilters import maximum_filter, minimum_filter
from spatialdata import SpatialData

from napari_sparrow.image._apply import ChannelList, apply


def min_max_filtering(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    size_min_max_filter: int | List[int] = 85,
    output_layer="min_max_filtered",
    overwrite: bool = False,
) -> SpatialData:
    """
    Apply min max filtering to an image in a SpatialData object using dask.
    The size of the filter can be provided
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
    overwrite: bool
        If True overwrites the element if it already exists.

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

    def _apply_min_max_filtering(image: Array, size_min_max_filter: int = 85) -> Array:
        # Apply the minimum filter
        minimum_t = minimum_filter(image, size_min_max_filter)

        # Apply the maximum filter
        max_of_min_t = maximum_filter(minimum_t, size_min_max_filter)

        return image - max_of_min_t

    if isinstance(size_min_max_filter, Iterable) and not isinstance(
        size_min_max_filter, str
    ):
        size_min_max_filter = ChannelList(size_min_max_filter)

    sdata = apply(
        sdata,
        _apply_min_max_filtering,
        img_layer=img_layer,
        output_layer=output_layer,
        chunks=None,
        channel=None,
        fn_kwargs={"size_min_max_filter": size_min_max_filter},
        overwrite=overwrite,
    )

    return sdata
