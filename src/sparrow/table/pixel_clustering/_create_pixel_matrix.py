from typing import Iterable

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from dask_image.ndfilters import gaussian_filter
from numpy.typing import NDArray
from spatialdata import SpatialData

from sparrow.image._image import _add_image_layer, _get_spatial_element
from sparrow.table._table import _add_table_layer
from sparrow.table.pixel_clustering._utils import _get_non_nan_pixel_values_and_location, _nonzeropercentile


def create_pixel_matrix(
    sdata: SpatialData,
    img_layer: str,  # should allow to specify a list of img_layers here to create the pixel matrix (case where you have multiple fov's).
    output_layer: str,
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    q: int | None = 99,
    q_norm_sum: int | None = 5,
    sigma: int | Iterable[int] | None = 2,  # set to 0 for specific channel to omit gaussian blur for specific channel
    fraction: float = 0.2,
    chunks: str | int | tuple[int, int] | None = None,  # chunks in y and x dimension if a tuple, for rechunking.
    seed: int = 10,
    overwrite: bool = False,
) -> SpatialData:
    assert 0 < fraction <= 1, "Value must be between 0 and 1"
    se_image = _get_spatial_element(sdata, layer=img_layer)

    if channels is not None:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]
    else:
        if channels is None:
            channels = se_image.c.data

    arr = se_image.sel(c=channels).data

    to_squeeze = False
    if arr.ndim == 3:
        # add trivial z dimension for 2D case
        arr = arr[:, None, ...]
        to_squeeze = True

    # fix chunk parameter
    if chunks is not None:
        if not isinstance(chunks, (int, str)):
            assert len(chunks) == arr.ndim - 2, "Please (only) provide chunks for ( 'y', 'x')."
            chunks = (arr.shape[0], arr.shape[1], chunks[0], chunks[1])
        elif isinstance(chunks, int):
            chunks = (arr.shape[0], arr.shape[1], chunks, chunks)

    if chunks is not None:
        arr = arr.rechunk(chunks)

    assert (
        arr.numblocks[0] == 1 and arr.numblocks[1] == 1
    ), "The number of blocks in 'c' and 'z' dimension should be equal to 1. Please specify the chunk parameter, to allow rechunking, e.g. `chunks=(1024,1024)`."

    # 1) calculate percentiles
    results_percentile = []
    for i in range(len(channels)):
        arr_percentile = _nonzeropercentile(arr[i], q=q)
        results_percentile.append(arr_percentile)
    arr_percentile = da.stack(results_percentile, axis=0)

    # 2) calculate norm sum percentile for img_layer
    # now normalize by percentile (percentile for each channel separate)
    arr_norm = arr / da.asarray(arr_percentile[..., None, None, None])

    # sum over all channels
    arr_norm_sum = da.sum(arr_norm, axis=0)
    # norm_sum_percentile = da.percentile(arr_norm_sum.flatten(), q=q_norm_sum)
    norm_sum_percentile = _nonzeropercentile(
        arr_norm_sum, q=q_norm_sum
    ).compute()  # pixel_thresh_val in ark analysis, if multiple images, we take average over all norm_sum_percentile for all images, and we use that value later on.
    # TODO also works without compute here, but then in adata, the uncomputed result is saved as a dask task graph, so maybe we should not save this value in adata as a workaround.

    # 3) gaussian blur
    if sigma is not None:
        sigma = list(sigma) if isinstance(sigma, Iterable) else [sigma] * len(channels)
        assert (
            len(sigma) == len(channels)
        ), f"If 'sigma' is provided as a list, it should match the number of channels in '{se_image}', or the number of channels provided via the 'channels' parameter '{channels}'."
        results_gaussian = []
        for _c_arr_norm, _sigma in zip(arr_norm, sigma):
            # run gaussian filter on each z stack independently, otherwise issues with depth
            results_gaussian_z = []
            for _z_c_arr_norm in _c_arr_norm:
                _z_c_arr_norm = gaussian_filter(_z_c_arr_norm, sigma=_sigma)
                results_gaussian_z.append(_z_c_arr_norm)
            results_gaussian.append(da.stack(results_gaussian_z, axis=0))
        arr_norm_gaussian = da.stack(results_gaussian, axis=0)
    else:
        arr_norm_gaussian = arr_norm

    arr_norm_gaussian_sum = da.sum(arr_norm_gaussian, axis=0)

    # sanity update to make sure that if pixel at certain location in a channel is nan, all pixels at that location for all channels are set to nan.
    arr_norm_gaussian = da.where(~da.isnan(arr_norm_gaussian_sum), arr_norm_gaussian, np.nan)

    # 4) normalize
    # set pixel values for which sum over all channels are below norm_sum_percentile to NaN
    arr_norm_gaussian = da.where(arr_norm_gaussian_sum > norm_sum_percentile, arr_norm_gaussian, np.nan)

    # recompute the sum (previous step put all pixel positions below threshold to nan), discard the nans for the sum
    arr_norm_gaussian_sum = da.nansum(arr_norm_gaussian, axis=0)
    # for each pixel position, divide by its sum over all channels, if sum is 0 (i.e. all channels give zero at this pixel position, set it to zero)
    arr_normalized = da.where(arr_norm_gaussian_sum > 0, arr_norm_gaussian / arr_norm_gaussian_sum, np.nan)

    # Now sample from the normalized pixel matrix.
    arr_normalized = da.map_blocks(
        _sampling_function,
        arr_normalized,
        seed=seed,
        fraction=fraction,
    )

    arr_sampled = _get_non_nan_pixel_values_and_location(arr_normalized)

    # create anndata object
    var = pd.DataFrame(index=channels)
    var.index.name = "channels"  # maybe add key
    var.index = var.index.map(str)

    adata = AnnData(X=arr_sampled[:, :-3], var=var)
    adata.uns[f"{img_layer}_percentile_{q_norm_sum}_norm_sum"] = norm_sum_percentile
    adata.var[f"{img_layer}_percentile_{q}"] = arr_percentile
    # add coordinates to anndata
    if to_squeeze:
        # 2D case
        adata.obsm["spatial"] = arr_sampled[:, -2:]
    else:
        # 3D case
        adata.obsm["spatial"] = arr_sampled[:, -3:]

    if to_squeeze:
        # arr_normalized is the sampled array
        arr_normalized = arr_normalized.squeeze(1)

    # sample in each chunk, only sample the non-nan values.

    sdata = _add_image_layer(
        sdata,
        arr=arr_normalized,
        output_layer=f"{img_layer}_gaussian",
        overwrite=True,
        c_coords=channels,
    )

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=None,
        overwrite=overwrite,
    )

    return sdata


def _sampling_function(block: NDArray, seed: int, fraction: float = 0.2) -> NDArray:
    # sampling by setting all values to np.nan that are not sampled.
    assert block.ndim == 4, "block should contain dimension (c, z, y, x)"
    rng = np.random.default_rng(seed=seed)

    # Find indices where array is not NaN -> valid indices to sample from
    valid_indices = np.argwhere(~np.isnan(block))

    valid_indices_channel_0 = valid_indices[valid_indices[:, 0] == 0][:, 1:]
    c_indices = np.unique(valid_indices[:, 0])

    # sanity check, to see if nan value are at same positions for all channels
    for c_index in c_indices:
        valid_indices_channel_c_index = valid_indices[valid_indices[:, 0] == c_index][:, 1:]
        assert np.array_equal(valid_indices_channel_0, valid_indices_channel_c_index)

    # Number of samples to draw (i.e. for each channel n_samples will be drawn), n_samples should be fraction of non nan values in chunk
    n_samples = int(fraction * (len(valid_indices_channel_0)))

    # Sample randomly among the valid indices
    sampled_indices = valid_indices_channel_0[rng.choice(len(valid_indices_channel_0), size=n_samples, replace=False)]

    # Create a mask with all elements set to False
    mask = np.zeros_like(block, dtype=bool)

    # Set True at sampled positions
    mask[:, sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]] = True

    # Set non-sampled values to nan
    result_array = np.where(mask, block, np.nan)

    return result_array
