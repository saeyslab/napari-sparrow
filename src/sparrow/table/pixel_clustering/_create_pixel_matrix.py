from typing import Any, Iterable

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array import Array
from dask_image.ndfilters import gaussian_filter
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t

from sparrow.image._image import _add_image_layer, _get_spatial_element, _get_transformation
from sparrow.table._table import _add_table_layer
from sparrow.table.pixel_clustering._utils import _nonzero_nonnan_percentile
from sparrow.utils._keys import _INSTANCE_KEY, _REGION_KEY


def create_pixel_matrix(
    sdata: SpatialData,
    img_layer: str | Iterable[str],
    output_table_layer: str,
    output_img_layer: str | Iterable[str] | None = None,
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    q: float | None = 99,
    q_sum: float | None = 5,
    q_post: float = 99.9,
    sigma: float | Iterable[float] | None = 2,
    norm_sum: bool = True,
    fraction: float = 0.2,
    chunks: str | int | tuple[int, int] | None = None,
    seed: int = 10,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Processes image layers specified in `img_layer` to create a pixel matrix and optionally normalizes and blurs the images based on various quantile and gaussian blur parameters. The results are added to `sdata` as specified in `output_table_layer` and optionally in `output_img_layer`.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the image data.
    img_layer : str | Iterable[str]
        The image layer(s) from `sdata` to process. This can be a single layer or a list of layers, e.g., when multiple fields of view are available.
    output_table_layer : str
        The name of the layer in `sdata.tables` where the resultant pixel matrix will be stored.
    output_img_layer : str | Iterable[str] | None, optional
        If specified, the preprocessed images are saved under this layer in `sdata`.
    channels : int | str | Iterable[int] | Iterable[str] | None, optional
        Specifies the channels to be included in the processing.
    q : float | None, optional
        Quantile used for normalization. If specified, pixel values are normalized by this quantile across the specified channels.
        Each channel is normalized by its own calculated quantile.
    q_sum : float | None, optional
        If the sum of the channel values at a pixel is below this quantile, the pixel values across all channels are set to NaN (and therefore excluded for being sampled).
    q_post : float, optional
        Quantile used for percentile calculations on the postprocessed channel matrix. Does not affect the resulting pixel matrix.
    sigma : float | Iterable[float] | None, optional
        Gaussian blur parameter for each channel. Use `0` to omit blurring for specific channels or `None` to skip blurring altogether.
    norm_sum : bool, optional
        If `True`, each channel is normalized by the sum of all channels at each pixel.
    fraction : float, optional
        Fraction of the data to sample for creation of the pixel matrix, useful for very large datasets.
    chunks : str | int | tuple[int, int] | None, optional
        Chunk sizes for processing. If provided as a tuple, these are the chunk sizes that will be used for rechunking in the y and x dimensions.
    seed : int, optional
        Seed for random operations, ensuring reproducibility when sampling data.
    scale_factors
        Scale factors to apply for multiscale. Ignored if `output_img_layer` is set to None.
    overwrite : bool, optional
        If `True`, overwrites existing data in the specified `output_table_layer` and `output_img_layer`.

    Notes
    -----
    To avoid data leakage:
     - in the single fov case, to prevent data leakage between channels, one should set `q_sum`==None and `norm_sum`==False, the only normalization that will be performed will then be a division by the `q` quantile value per channel.
     - in the multiple fov case, both `q_sum`, `norm_sum` and `q` should be set to None to prevent data leakage both between channels and between images. In the multiple fov case one could opt for creating the pixel matrix for each fov individually, and then merge the resulting tables.

    Returns
    -------
    SpatialData
        An updated SpatialData object with the newly created pixel matrix and optionally preprocessed image data stored in specified layers.
    """
    # setting q_sum =None, and norm_sum=False -> then there will be no data leakage in single fov case.
    assert 0 < fraction <= 1, "Value must be between 0 and 1"
    img_layer = list(img_layer) if isinstance(img_layer, Iterable) and not isinstance(img_layer, str) else [img_layer]
    if output_img_layer is not None:
        output_img_layer = (
            list(output_img_layer)
            if isinstance(output_img_layer, Iterable) and not isinstance(output_img_layer, str)
            else [output_img_layer]
        )
        assert len(output_img_layer) == len(
            img_layer
        ), "The number of `output_img_layer` specified should be the equal to the the number of `img_layer` specified."

    se_image = _get_spatial_element(sdata, layer=img_layer[0])

    if channels is not None:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]
    else:
        if channels is None:
            channels = se_image.c.data

    _arr_list = []
    _transformations = []
    for i, _img_layer in enumerate(img_layer):
        se_image = _get_spatial_element(sdata, layer=_img_layer)
        _transformations.append(_get_transformation(se_image))
        arr = se_image.sel(c=channels).data
        if i == 0:
            _array_dim = arr.ndim
        else:
            assert (
                _array_dim == arr.ndim
            ), "Image layers specified via parameter `img_layer` should all have same number of dimensions."

        to_squeeze = False
        if arr.ndim == 3:
            # add trivial z dimension for 2D case
            arr = arr[:, None, ...]
            to_squeeze = True
        _arr_list.append(arr)

    for i in range(len(_arr_list)):
        # fix chunk parameter
        if chunks is not None:
            if not isinstance(chunks, (int, str)):
                assert len(chunks) == _arr_list[i].ndim - 2, "Please (only) provide chunks for ( 'y', 'x')."
                _chunks = (_arr_list[i].shape[0], _arr_list[i].shape[1], chunks[0], chunks[1])
            elif isinstance(chunks, int):
                _chunks = (_arr_list[i].shape[0], _arr_list[i].shape[1], chunks, chunks)
            elif isinstance(chunks, str):
                _chunks = chunks

        if chunks is not None:
            _arr_list[i] = _arr_list[i].rechunk(_chunks)

        assert (
            _arr_list[i].numblocks[0] == 1 and _arr_list[i].numblocks[1] == 1
        ), "The number of blocks in 'c' and 'z' dimension should be equal to 1. Please specify the chunk parameter, to allow rechunking, e.g. `chunks=(1024,1024)`."

    if q is not None:
        results_arr_percentile = []
        for _arr in _arr_list:
            # 1) calculate percentiles (excluding nan and 0 from calculation)
            arr_percentile = _nonzero_nonnan_percentile_axis_0(_arr, q=q)
            results_arr_percentile.append(arr_percentile)
        arr_percentile = da.stack(results_arr_percentile, axis=0)
        arr_percentile_mean = da.mean(arr_percentile, axis=0)  # mean over all images
        # for multiple img_layer, in ark one uses np.mean( arr_percentile ) as the percentile to normalize

    # 2) calculate norm sum percentile for img_layer
    # now normalize by percentile (percentile for each channel separate)
    if q is not None:
        for i in range(len(_arr_list)):
            _arr_list[i] = _arr_list[i] / da.asarray(arr_percentile_mean[..., None, None, None])

    # sum over all channels for each image
    _arr_sum_list = []
    for _arr in _arr_list:
        _arr_sum_list.append(da.sum(_arr, axis=0))
    # norm_sum_percentile = da.percentile(arr_norm_sum.flatten(), q=q_norm_sum)
    # in ark_analysis, np.quantile is used, which uses 0's for quantile computation, so equivalent would be da.percentile, not sure if we should also use it instead of _nonzeropercentile
    if q_sum is not None:
        results_norm_sum_percentile = []
        for _arr_sum in _arr_sum_list:
            # using da.percentile reproduces exactly results of ark, but nonzero_nonnan feels like a better choice (i.e. case where there is a lot of zero in image)
            # norm_sum_percentile = da.percentile(_arr_sum.flatten(), q=q_sum)
            norm_sum_percentile = _nonzero_nonnan_percentile(_arr_sum, q=q_sum)
            results_norm_sum_percentile.append(norm_sum_percentile)
        # pixel_thresh_val in ark analysis, if multiple images, we take average over all norm_sum_percentile for all images, and we use that value later on.
        norm_sum_percentile = da.stack(results_norm_sum_percentile, axis=0)
        norm_sum_percentile = da.mean(norm_sum_percentile, axis=0)

    # 3) gaussian blur
    if sigma is not None:
        sigma = list(sigma) if isinstance(sigma, Iterable) else [sigma] * len(channels)
        assert (
            len(sigma) == len(channels)
        ), f"If 'sigma' is provided as a list, it should match the number of channels in '{se_image}', or the number of channels provided via the 'channels' parameter '{channels}'."
        # gaussian blur for each image separately
        for i in range(len(_arr_list)):
            _arr_list[i] = _gaussian_blur(_arr_list[i], sigma=sigma)
        # recompute the sum over all channels
        _arr_sum_list = []
        for _arr in _arr_list:
            _arr_sum_list.append(da.sum(_arr, axis=0))

    # sanity check
    assert len(_arr_sum_list) == len(_arr_list) == len(img_layer)
    results_arr_sampled = []
    results_arr_percentile_post_norm = []
    _region_keys = []
    for i in range(len(_arr_list)):
        # sanity update to make sure that if pixel at certain location in a channel is nan, all pixels at that location for all channels are set to nan.
        # use fact that sum is nan if one of the values is nan along that axis -> this is necesarry for map_blocks of the _sampling function
        _arr_list[i] = da.where(~da.isnan(_arr_sum_list[i]), _arr_list[i], np.nan)

        # 4) normalize
        # set pixel values for which sum over all channels are below norm_sum_percentile to NaN
        if q_sum is not None:
            _arr_list[i] = da.where(_arr_sum_list[i] > norm_sum_percentile, _arr_list[i], np.nan)
        if norm_sum:
            # recompute the sum (previous step puts all pixel positions below threshold to nan), discard the nans for the sum
            _arr_sum = da.nansum(_arr_list[i], axis=0)
            # for each pixel position, divide by its sum over all channels, if sum is 0 (i.e. if all channels give zero at this pixel position, set it to nan)
            _arr_list[i] = da.where(_arr_sum > 0, _arr_list[i] / _arr_sum, np.nan)

        arr_percentile_post_norm = _nonzero_nonnan_percentile_axis_0(_arr_list[i], q=q_post)
        results_arr_percentile_post_norm.append(arr_percentile_post_norm)

        # save the preprocessed images, in this way we get the preprocessed images from which we sample
        if output_img_layer is not None:
            sdata = _add_image_layer(
                sdata,
                arr=_arr_list[i].squeeze(1) if to_squeeze else _arr_list[i],
                output_layer=output_img_layer[i],
                transformation=_transformations[i],
                scale_factors=scale_factors,
                c_coords=channels,
                overwrite=overwrite,
            )

            se_output_image = _get_spatial_element(sdata, layer=output_img_layer[i])
            _arr_list[i] = (
                se_output_image.sel(c=channels).data[:, None, ...]
                if to_squeeze
                else se_output_image.sel(c=channels).data
            )

        # Now sample from the normalized pixel matrix.
        chunksize = _arr_list[i].chunksize
        # rechunk to avoid needing to sample for every channel seperately
        _arr_list[i] = _arr_list[i].rechunk((_arr_list[i].shape[0], _arr_list[i].shape[1], chunksize[2], chunksize[3]))
        chunksize = _arr_list[i].chunksize

        # this creates a dask array of shape (1,1, int(fraction * np.prod(_arr_list[i].chunksize[1:]))*nr_of_chunks_y, (nr_of_channels +3)*nr_of_chunks_x )
        output_chunksize = (1, 1, int(fraction * np.prod(chunksize[1:])), chunksize[0] + 3)
        _arr_list[i] = da.map_blocks(
            _sampling_function,
            _arr_list[i],
            seed=seed,
            dtype=np.float32,
            fraction=fraction,
            chunks=output_chunksize,
            _chunksize=chunksize,
        )

        # compute the sampling
        _arr_sampled = _arr_list[i].squeeze((0, 1)).compute()
        # reshape back to array of shape num_samples * (num_channels + 3)
        _arr_sampled = _reshape(_arr_sampled, output_chunksize[2:])
        results_arr_sampled.append(_arr_sampled)
        _region_keys.extend(_arr_sampled.shape[0] * [img_layer[i]])

    arr_sampled = np.row_stack(results_arr_sampled)
    arr_percentile_post_norm = da.stack(results_arr_percentile_post_norm, axis=0)

    # create anndata object
    var = pd.DataFrame(index=channels)
    var.index.name = "channels"
    var.index = var.index.map(str)

    adata = AnnData(X=arr_sampled[:, :-3], var=var)

    if q is not None:
        for _img_layer, _arr_percentile in zip(img_layer, arr_percentile):
            adata.var[f"{_img_layer}_percentile_{q}"] = _arr_percentile
        adata.var[f"mean_percentile_{q}"] = arr_percentile_mean
    for _img_layer, _arr_percentile_post_norm in zip(img_layer, arr_percentile_post_norm):
        adata.var[f"{_img_layer}_post_norm_percentile_{q_post}"] = _arr_percentile_post_norm
    # use this one to normalize before pixel clustering
    adata.var[f"mean_post_norm_percentile_{q_post}"] = da.mean(arr_percentile_post_norm, axis=0)

    adata.obs[_INSTANCE_KEY] = np.arange(arr_sampled.shape[0])
    adata.obs[_REGION_KEY] = pd.Categorical(_region_keys)  # should it also be possible to link a labels layer?

    # add coordinates to anndata
    if to_squeeze:
        # 2D case, only save y,x position
        adata.obsm["spatial"] = arr_sampled[:, -2:]
    else:
        # 3D case, save z,y,x position
        adata.obsm["spatial"] = arr_sampled[:, -3:]

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_table_layer,
        region=img_layer,
        overwrite=overwrite,
    )

    return sdata


def _sampling_function(
    block: NDArray, fraction: float, seed: int, _chunksize: tuple[int, int, int, int], block_info: dict[str:Any]
) -> NDArray:
    def _sample_values_and_z_y_x_indices(arr: NDArray, num_samples: int, seed: int = 20):
        assert arr.ndim == 4

        rng = np.random.default_rng(seed=seed)

        total_size = np.prod(arr.shape[1:])
        flat_indices = rng.choice(total_size, size=num_samples, replace=False)

        # Convert flat indices to 3D indices
        z_indices, y_indices, x_indices = np.unravel_index(flat_indices, arr.shape[1:])

        sampled_values = arr[:, z_indices, y_indices, x_indices]

        stacked = np.row_stack(
            [sampled_values, z_indices.reshape(1, -1), y_indices.reshape(1, -1), x_indices.reshape(1, -1)]
        )

        # return a numpy array with shape (1,1,num_samples, arr.shape[0]+1+1+1 )
        return stacked.T[None, None, ...]

    assert block.ndim == 4
    chunk_location = block_info[0]["chunk-location"]
    offset_z = block_info[0]["array-location"][1][0]
    offset_y = block_info[0]["array-location"][2][0]
    offset_x = block_info[0]["array-location"][3][0]

    chunk_shape = block.shape

    # unique chunk ID for given z,y,x location.
    chunk_id = (
        chunk_location[1] * (chunk_shape[2] * chunk_shape[3]) + chunk_location[2] * chunk_shape[3] + chunk_location[3]
    )
    # make seed unique for each chunk in z,y,x so we sample differently in each of these chunks, but sample the same for given chunk in z,y,x in different channels.
    seed = seed + chunk_id

    total_size_z_y_x = np.prod(block.shape[1:])
    num_samples = int(fraction * total_size_z_y_x)
    if num_samples == 0:
        num_samples = 1

    block = _sample_values_and_z_y_x_indices(block, num_samples=num_samples, seed=seed)
    # add the offset
    block[:, :, :, -3] = block[:, :, :, -3] + offset_z
    block[:, :, :, -2] = block[:, :, :, -2] + offset_y
    block[:, :, :, -1] = block[:, :, :, -1] + offset_x

    # pad here so we get same output block dimensions for every chunk, remove nans in later step.
    padded_total_size_z_y_x = int(fraction * np.prod(_chunksize[1:]))
    if padded_total_size_z_y_x == 0:
        padded_total_size_z_y_x = 1

    padding_y = padded_total_size_z_y_x - block.shape[2]
    if padding_y > 0:
        pad_width = [(0, 0), (0, 0), (0, padding_y), (0, 0)]  # No padding for c, z, x dimensions, padding for y
        block = np.pad(block, pad_width=pad_width, mode="constant", constant_values=np.nan)

    return block


def _reshape(array: NDArray, chunksize: tuple[int, int]) -> NDArray:
    assert array.ndim == 2
    assert len(chunksize) == 2

    split_rows = np.split(array, indices_or_sections=array.shape[0] // chunksize[0])
    tiles = [
        np.split(row_block, indices_or_sections=row_block.shape[1] // chunksize[1], axis=1) for row_block in split_rows
    ]

    array = np.vstack([tile for sublist in tiles for tile in sublist])

    # remove sampels for which all channels are NaN
    nan_mask = np.isnan(array[:, :-3]).all(axis=1)

    return array[~nan_mask]


def _nonzero_nonnan_percentile_axis_0(arr: Array, q: float):
    results_percentile = []
    for i in range(arr.shape[0]):
        arr_percentile = _nonzero_nonnan_percentile(arr[i], q=q)
        results_percentile.append(arr_percentile)
    return da.stack(results_percentile, axis=0)


def _gaussian_blur(
    arr: Array,
    sigma: list[float],
) -> Array:
    results_gaussian = []
    for _c_arr_norm, _sigma in zip(arr, sigma):
        # run gaussian filter on each z stack independently, otherwise issues with depth when having few z-stacks in gaussian_filter
        results_gaussian_z = []
        for _z_c_arr_norm in _c_arr_norm:
            _z_c_arr_norm = gaussian_filter(_z_c_arr_norm, sigma=_sigma)
            results_gaussian_z.append(_z_c_arr_norm)
        results_gaussian.append(da.stack(results_gaussian_z, axis=0))
    return da.stack(results_gaussian, axis=0)
