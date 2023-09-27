from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional

import dask.array as da
import spatialdata
from dask.array import Array
from dask.array.overlap import coerce_depth
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation, set_transformation

from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def apply(
    sdata: SpatialData,
    func: Callable[..., NDArray | Array],
    img_layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    channel: Optional[int | Iterable[int]] = None,
    chunks: str | tuple[int, int] | int | None = None,
    overwrite: bool = False,
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> SpatialData:
    """
    Apply a specified function to an image layer in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the image to be processed.
    func : Callable[..., NDArray | Array]
        The Callable to apply to the image.
    img_layer : Optional[str], default=None
        The image layer in `sdata` to process. If not provided, the last image layer in `sdata` is used.
    output_layer : Optional[str]
        The name of the output layer where results will be stored. This must be specified.
    channel : Optional[int | Iterable[int]], default=None
        Specifies which channel(s) to run `func` on. The `func` is run independently on each channel.
          If None, the `func` is run on all channels.
    chunks : str | tuple[int, int] | int | None, default=None
        Specification for rechunking the data before applying the function.
        If specified, dask's map_overlap or map_blocks is used depending on the occurence of the "depth" parameter in kwargs.
    overwrite : bool, default=False
        If True, overwrites the output layer if it already exists in `sdata`.
    fn_kwargs : Mapping[str, Any], default=MappingProxyType({})
        Keyword arguments to pass to `func`.
        If a different value should be used for each channel, then value to the keyword argument
        should be passed as a ChannelList with length equal to the number of channels in the image.
    **kwargs : Any
        Additional keyword arguments passed to dask's map_overlap or map_blocks
        depending of the occurence of "depth" in kwargs.

    Returns
    -------
    SpatialData
        The `sdata` object with the processed image added to the specified output layer.

    Raises
    ------
    ValueError
        If `output_layer` is not provided.
        If ChannelList objects are specified in fn_kwargs that do not match the provided number of channels.
        If provided arrays are not 2D (c,y,x).

    Notes
    -----
    This function is designed for processing images stored in a SpatialData object using dask for potential
    parallelism and out-of-core computation. Depending on the `chunks` parameter and other settings,
    it can leverage dask's map_overlap or map_blocks functions to apply the desired image processing function.

    Examples
    --------
    Apply a custom function `my_function` to all channels in an image using different parameters for each channel
    (we assume sdata[ "raw_image" ] has 2 channels):

    >>> def my_function( image, parameter ):
    ...    return image*parameter
    >>> fn_kwargs={ "parameter": ChannelList( [2,3] )  }
    >>> sdata = apply(sdata, my_function, img_layer="raw_image", output_layer="processed_image", channel=None, fn_kwargs=fn_kwargs )

    Apply the same function to only the first channel of the image:

    >>> fn_kwargs={ "parameter": 2 }
    >>> sdata = apply(sdata, my_function, img_layer="raw_image", output_layer="processed_image", channel=0, fn_kwargs=fn_kwargs )
    """

    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Applying image processing on the last image layer '{img_layer}' of the provided SpatialData object."
        )

    if output_layer is None:
        raise ValueError("Please specify a name for the output layer.")

    def apply_func(
        func: Callable[..., NDArray | Array],
        arr: NDArray | Array,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> Array:
        if chunks is None:
            arr = func(arr, **fn_kwargs)
            return da.asarray(arr)
        arr = da.asarray(arr).rechunk(chunks)
        if "depth" in kwargs:
            kwargs.setdefault("boundary", "reflect")

            # Adjusting depth here prevents rechunking in map_overlap via the ensure_minimum_chunksize function.
            # This way we do not have unexpected chunk sizes/memory use as result.
            def adjust_depth(depth, chunksize, depth_dim):
                original_depth = depth[depth_dim]
                if chunksize[depth_dim] < original_depth:
                    depth[depth_dim] = chunksize[depth_dim] // 4
                    log.warning(
                        f"The overlapping depth '{original_depth}' is larger than your chunksize '{chunksize[depth_dim]}' along dimension '{depth_dim}'. "
                        f"Setting depth to 'chunks//4 ({depth[depth_dim]})'"
                    )

            chunksize = arr.chunksize
            depth = kwargs["depth"]
            depth = coerce_depth(arr.ndim, depth)

            for dim in range(arr.ndim):
                adjust_depth(depth, chunksize, dim)

            kwargs["depth"] = depth

            arr = da.map_overlap(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        else:
            arr = da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        return arr.rechunk(chunks)

    if channel is not None:
        channel = (
            list(channel)
            if isinstance(channel, Iterable) and not isinstance(channel, str)
            else [channel]
        )
    else:
        channel = sdata[img_layer].c.data

    # create fn_kwargs for each channel

    for key, value in list(fn_kwargs.items()):
        if isinstance(value, ChannelList) and len(value) != len(channel):
            raise ValueError(
                f"The value of parameter '{key}' is a ChannelList ({value}), it must have a length equal to the number of channels ({channel})."
            )
        elif not isinstance(value, ChannelList):
            fn_kwargs[key] = ChannelList([value] * len(channel))

    _fn_kwargs_channel = [
        {k: v[i] for k, v in fn_kwargs.items()}
        for i in range(len(next(iter(fn_kwargs.values()))))
    ]

    # sanity ceck
    assert len(_fn_kwargs_channel) == len(channel)

    # store results per channel
    results = []

    for ch, _fn_kwargs in zip(channel, _fn_kwargs_channel):
        arr = sdata[img_layer].isel(c=ch).data
        if len(arr.shape) != 2:
            raise ValueError(
                f"Array is of dimension {arr.shape}, currently only 2D images are supported."
            )
        # need to pass correct value from fn_kwargs to apply_func
        arr = apply_func(func, arr, _fn_kwargs)
        results.append(arr)

    arr = da.stack(results, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(arr, dims=("c", "y", "x"))

    # TODO maybe also make it possible to send transformation with the apply function
    # now by default we copy transformation of old img_layer to new img_layer
    trf = get_transformation(sdata[img_layer])
    set_transformation(spatial_image, trf)

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image, overwrite=overwrite)

    return sdata


class ChannelList:
    def __init__(self, *args, **kwargs):
        self._list = list(*args, **kwargs)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        return self._list[index]

    def __setitem__(self, index, value):
        self._list[index] = value

    def __delitem__(self, index):
        del self._list[index]

    def append(self, item):
        self._list.append(item)

    def extend(self, items):
        self._list.extend(items)

    def remove(self, item):
        self._list.remove(item)

    def __repr__(self):
        return repr(self._list)
