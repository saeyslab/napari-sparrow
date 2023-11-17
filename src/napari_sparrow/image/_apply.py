from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import dask.array as da
from dask.array import Array
from dask.array.overlap import coerce_depth
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from napari_sparrow.image._image import (
    _get_spatial_element,
    _get_translation,
    _substract_translation_crd,
    _add_image_layer,
)
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def apply(
    sdata: SpatialData,
    func: Callable[..., NDArray | Array],
    img_layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    channel: Optional[int | Iterable[int]] = None,
    z_slice: Optional[int | Iterable[int]] = None,
    combine_c=False,
    combine_z=True,
    chunks: Optional[str | int | Tuple[int, ...]] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
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
    chunks : str | Tuple[int, ...] | int | None, default=None
        Specification for rechunking the data before applying the function.
        If specified, dask's map_overlap or map_blocks is used depending on the occurence of the "depth" parameter in kwargs.
        If chunks is a Tuple, they  contain the chunk size that will be used in the spatial dimensions.
    crd : Optional[Tuple[int, int, int, int]], default=None
        The coordinates specifying the region of the image to be processed. Defines the bounds (x_min, x_max, y_min, y_max).
    scale_factors
        Scale factors to apply for multiscale.
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
    >>> fn_kwargs={ "parameter": ChannelList( [2,3] ) }
    >>> sdata = apply(sdata, my_function, img_layer="raw_image", output_layer="processed_image", channel=None, fn_kwargs=fn_kwargs)

    Apply the same function to only the first channel of the image:

    >>> fn_kwargs={ "parameter": 2 }
    >>> sdata = apply(sdata, my_function, img_layer="raw_image", output_layer="processed_image", channel=0, fn_kwargs=fn_kwargs)
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
            # if dask array, we want to rechunk,
            # because taking a crop could have caused irregular chunks
            if isinstance(arr, Array):
                arr = arr.rechunk(arr.chunksize)
            arr = func(arr, **fn_kwargs)
            return da.asarray(arr)
        if not isinstance(chunks, (int, str)):
            if len(chunks) != arr.ndim:
                raise ValueError(
                    f"Chunks ({chunks}) are provided for {len(chunks)} dimensions. "
                    f"Please (only) provide chunks for the {arr.ndim} spatial dimensions."
                )
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
            if not isinstance(depth, int):
                if len(depth) != arr.ndim:
                    raise ValueError(
                        f"Depth ({depth}) is provided for {len(depth)} dimensions. "
                        f"Please (only) provide depth for the {arr.ndim} spatial dimensions."
                    )
            depth = coerce_depth(arr.ndim, depth)

            for dim in range(arr.ndim):
                adjust_depth(depth, chunksize, dim)

            kwargs["depth"] = depth

            arr = da.map_overlap(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        else:
            arr = da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        return arr.rechunk(chunks)

    # get spatial element
    se = _get_spatial_element(sdata, layer=img_layer)

    # here you specify the channels the function is applied on
    if channel is not None:
        channel = (
            list(channel)
            if isinstance(channel, Iterable) and not isinstance(channel, str)
            else [channel]
        )
    else:
        channel = se.c.data

    # here you specify the z-stacks the function is applied on
    if z_slice is not None:
        # TODO check if this makes sense, i.e. se should be 3 dimensional
        z_slice = (
            list(z_slice)
            if isinstance(z_slice, Iterable) and not isinstance(z_slice, str)
            else [z_slice]
        )
    else:
        z_slice = se.z.data

    # sanity check on 3D. If no z-dimension, set combine_z to True
    if 'z' not in se.dims:
        # set combine_z trivial to True if there is no z dimension
        combine_z=True

    fn_kwargs, func=_precondition( fn_kwargs=fn_kwargs, func=func, combine_c=combine_c, combine_z=combine_z, channels=channel, z_slices=z_slice )

    if not combine_c:
        # so process them all indepentenly
        # TODO not allow is se is not 3D (fix this by setting combine_z to True if se does not contain z dimensions)
        if not combine_z:
            result=[]
            for key_c, _fn_kwargs_c in fn_kwargs.items():
                func_c=func[ key_c ]
                result_z=[]
                for key_z, _fn_kwargs_c_z in _fn_kwargs_c.items():
                    func_c_z=func_c[ key_z ]
                    arr=se.sel( z=[key_z] , c=[key_c] ).data  # we do not reduce dimensions, we do sel with z=[...]
                    if crd is not None:
                        arr = arr[ : ,:, crd[2] : crd[3], crd[0] : crd[1]]
                    arr=apply_func( func=func_c_z, arr=arr, fn_kwargs=_fn_kwargs_c_z )
                    # arr of size (1,1,y,x)
                    result_z.append( arr[ 0, 0, ... ] )
                result.append(da.stack( result_z, axis=0 ))
            arr=da.stack( result, axis=0 )
        else:
            # this combines z. Thus all z dimensions are send to apply func
            result=[]
            for key_c, _fn_kwargs_c in fn_kwargs.items():
                func_c=func[ key_c ]
                arr=se.sel( c=[key_c] ).data  # we do not reduce dimensions, we do sel with c=[...]
                if crd is not None:
                    if arr.ndim == 3:
                        arr = arr[ : , crd[2] : crd[3], crd[0] : crd[1]]
                    elif arr.ndim ==4:
                        arr = arr[ : , :, crd[2] : crd[3], crd[0] : crd[1]]
                arr=apply_func( func=func_c, arr=arr, fn_kwargs=_fn_kwargs_c ) # apply func expects c,z,y,x
                # this returns 1, z, y, x, so we take z,y,x and do a stack over all results to get our c-dimension back
                result.append( arr[ 0, ... ] )
            arr=da.stack( result, axis=0 )
    else:
        if not combine_z:
            # TODO not allow is se is not 3D (fix this by setting combine_z to True if se does not contain z dimensions)
            result=[]
            for key_z, _fn_kwargs_z in fn_kwargs.items():
                func_z=func[ key_z ]
                arr=se.sel( c=[key_z] ).data
                if crd is not None:
                    arr = arr[ : ,:, crd[2] : crd[3], crd[0] : crd[1]]
                arr=apply_func( func=func_z, arr=arr, fn_kwargs=_fn_kwargs_z )

        else: # case where combine_z is True, and combine_c is True 
            # -> just regular fn_kwargs and func passed to apply, but on channels that are specified
            arr=se.sel( c=[channel], z=[z_slice] ).data
            if crd is not None:
                if arr.ndim == 3:
                    arr = arr[ : , crd[2] : crd[3], crd[0] : crd[1]]
                elif arr.ndim ==4:
                    arr = arr[ : , :, crd[2] : crd[3], crd[0] : crd[1]]
            arr=apply_func( func=func, arr=arr, fn_kwargs=fn_kwargs )

    if crd is not None:
        crd = _substract_translation_crd(se, crd)

    tx, ty = _get_translation(se)

    if crd is not None:
        tx = tx + crd[0]
        ty = ty + crd[2]

    translation = Translation([tx, ty], axes=("x", "y"))

    sdata = _add_image_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=arr.chunksize,
        transformation=translation,
        scale_factors=scale_factors,
        c_coords=channel,
        overwrite=overwrite,
    )

    return sdata


def _precondition( fn_kwargs, func, combine_c, combine_z, channels, z_slices ):

    def collect_keys_fn_kwargs(d):
        keys = []
        for key, value in d.items():
            keys.append(key)
            if isinstance(value,  (dict) ):
                result=collect_keys_fn_kwargs( value )
                if result is not None:
                    keys.extend( result )
            else:
                return
        return keys

    def collect_keys_func(d):
        keys = []
        for key, value in d.items():
            keys.append(key)
            if isinstance(value,  (dict) ):
                result=collect_keys_func( value )
                if result is not None:
                    keys.extend( result )
            else:
                if not isinstance( value, Callable ):
                    raise ValueError( "Should specify callable if a dict is provided." )
        return keys

    def flatten_list(nested_list):
        result = []
        for element in nested_list:
            if isinstance(element, list):
                result.extend(flatten_list(element))
            else:
                result.append(element)
        return result


    def make_mapping(keys, value):
        result = {}
        for item in keys:
            if isinstance(item, list):
                sub_dict = {k: value for k in item}
                result[previous_item] = sub_dict
            else:
                result[item] = value
                previous_item = item
        return result


    keys_fn_kwargs=collect_keys_fn_kwargs( fn_kwargs )
    if isinstance( func, Callable  ):
        if keys_fn_kwargs is not None:
            # if callable, make a mapping of it
            func=make_mapping( keys_fn_kwargs, func )

    # sanity checks
    if keys_fn_kwargs is not None:
        keys_func=collect_keys_func( func )
        assert keys_fn_kwargs == keys_func, "should specify same keys in fn_kwargs and func"
        keys=set(flatten_list( keys_fn_kwargs ))
        # now also do sanity check on the channels and z slices.
        if combine_c:
            if keys.intersection( set(channels) ):
                raise ValueError( "keys in fn_kwargs also present in channels can not be specified if process_each_channel_indepentenlyt is set to false." )
        if combine_z:
            if keys.intersection( set(z_slices) ):
                raise ValueError( "keyword argumens for z_slices can not be specified if process_each_z_stack_indepentenlyt is set to false." )

    # fix keys
    if not combine_c:
        if not set(fn_kwargs.keys()).issubset( set( channels ) ):
            if not combine_z:
                if not set(fn_kwargs.keys()).issubset( set( z_slices ) ):
                    # in this case func should be a callable
                    #if not isinstance( func, Callable ): 
                    #    raise ValueError
                    print( "process each zstack indepentenlty is True, but not all z-slices spefified in fn_kwargs/func." )
                    fn_kwargs={key: fn_kwargs for key in z_slices }
                    func={key: func for key in z_slices }
            fn_kwargs={key: fn_kwargs for key in channels } 
            func={key: func for key in channels }
            
        # case where we are subset of channels, but want to add z dimension
        elif not combine_z:
            for item, value in fn_kwargs.items():
                if not set(value.keys()).issubset( set( z_slices ) ):
                    print( "process each zstack indepentenlty is True, but not all z-slices spefified in fn_kwargs/func" )
                    # print a warning
                    fn_kwargs = {key: {z_slice: _value for z_slice in z_slices} for key, _value in fn_kwargs.items()}
                    func = {key: {z_slice: _value for z_slice in z_slices} for key, _value in func.items()}
                    break
    elif not combine_z:
        print( "process each zstack indepentenlty is True, but not all z-slices spefified in fn_kwargs/func" )
        if not set(fn_kwargs.keys()).issubset( set( z_slices ) ):
            fn_kwargs={key: fn_kwargs for key in z_slices }
            func={key: func for key in z_slices }
    else:
        assert isinstance( func, Callable )
        assert keys_fn_kwargs is None, "process_each_channel_indepetnently and process_each_z_stack_indepe are both set to False, but it seems fn_kwargs contains specific values for channels and z stakcs"
        
    return fn_kwargs, func



'''
def _precondition( fn_kwargs, func ):

    # sanity check:
    keys = set()
    for key, value in fn_kwargs.items():
        keys.add(key)
        if isinstance(value, dict):
            keys.update(value.keys())

    if not process_each_channel_indepentenly:
        if keys.issubset( set(channels) ):
            raise ValueError( "keys in fn_kwargs also present in channels can not be specified if process_each_channel_indepentenlyt is set to false." )

    if not process_each_z_stack_indepentenly:
        if keys.issubset( set(z_slices) ):
            raise ValueError( "keyword argumens for z_slices can not be specified if process_each_z_stack_indepentenlyt is set to false." )



    # fix keys
    if process_each_channel_indepentenly:
        if not set(fn_kwargs.keys()).issubset( set( channels ) ):
            if process_each_z_stack_indepentenly:
                if not set(fn_kwargs.keys()).issubset( set( z_slices ) ):
                    # in this case func should be a callable
                    #if not isinstance( func, Callable ): 
                    #    raise ValueError
                    print( "process each zstack indepentenlty is True, but not all z-slices spefified in fn_kwargs/func." )
                    fn_kwargs={key: fn_kwargs for key in z_slices }
                    func={key: func for key in z_slices }
                    #fn_kwargs={key: fn_kwargs for key in channels } 
                    #func={key: func for key in channels }
            fn_kwargs={key: fn_kwargs for key in channels } 
            func={key: func for key in channels }
            
        # case where we are subset of channels, but want to add z dimension
        elif process_each_z_stack_indepentenly:
        for item, value in fn_kwargs.items():
            if not set(value.keys()).issubset( set( z_slices ) ):
                print( "process each zstack indepentenlty is True, but not all z-slices spefified in fn_kwargs/func" )
                # print a warning
                fn_kwargs = {key: {z_slice: _value for z_slice in z_slices} for key, _value in fn_kwargs.items()}
                func = {key: {z_slice: _value for z_slice in z_slices} for key, _value in func.items()}
                break
            

    elif process_each_z_stack_indepentenly:

        print( "process each zstack indepentenlty is True, but not all z-slices spefified in fn_kwargs/func" )
        if not set(fn_kwargs.keys()).issubset( set( z_slices ) ):
            fn_kwargs={key: fn_kwargs for key in z_slices }
            func={key: func for key in z_slices }

    else:
        pass
'''