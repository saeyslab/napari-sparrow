from __future__ import annotations

from typing import Iterable, Optional

import dask.array as da
import numpy as np
import pandas as pd
import spatialdata
from anndata import AnnData
from dask import delayed
from dask.array import Array, unique
from numpy.typing import NDArray
from spatialdata import SpatialData

from napari_sparrow.image._image import _get_spatial_element, _get_translation
from napari_sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate_intensity(
    sdata: SpatialData,
    img_layer: Optional[str] = None,
    labels_layer: Optional[str] = None,
    channels: Optional[int | str | Iterable[int] | Iterable[str]] = None,
    chunks: Optional[str | int | tuple[int, ...]] = 10000,
    append: bool = False,
    append_labels_layer_name: bool = True,
) -> SpatialData:
    """
    Allocates intensity values from a specified image layer to corresponding cells in a SpatialData object and
    returns an updated SpatialData object with an attached table attribute containing the AnnData object with intensity values
    for each cell and each (specified) channel.

    It requires that the image layer and the labels layer have the same shape and alignment.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing spatial information about cells.
    img_layer : str, optional
        The name of the layer in `sdata` that contains the image data from which to extract intensity information.
        Both the `img_layer` and `labels_layer` should have the same shape and alignment. If not provided,
        will use last img_layer.
    labels_layer : str, optional
        The name of the layer in `sdata` containing the labels (segmentation) used to define the boundaries of cells.
        These labels correspond with regions in the `img_layer`. If not provided, will use last labels_layer.
    channels : int or str or Iterable[int] or Iterable[str], optional
        Specifies the channels to be considered when extracting intensity information from the `img_layer`.
        This parameter can take a single integer or string or an iterable of integers or strings representing specific channels.
        If set to None (the default), intensity data will be aggregated from all available channels within the image layer.
    chunks : str | int | tuple[int, ...], optional
        The chunk size for processing the image data.
    append: bool, optional.
        If set to True, the intensity values extracted during the current function call will be appended to any existing intensity data
        within the SpatialData object's table attribute. If False, any existing data in `sdata.table` will be overwritten by the newly extracted intensity values.
    append_labels_layer_name: bool, optional.
        Determines whether the labels layer name is added as suffix to the channel names in the resulting AnnData object at `sdata.table`.
        This is especially relevant when multiple label layers are used, helping keep the origin of the data clear.
        When set to True, the channel names in the 'var' DataFrame of `sdata.table` will be in the format "{channel}_{labels_layer}".
        For example, if dealing with a labels layer named "layer1", and channel "channelA", the resulting channel name in `sdata.table` would be "channelA_layer1".
        If False, the original channel names are preserved without modification.

    Returns
    -------
    SpatialData
        An updated version of the input SpatialData object. The updated object includes a new 'table' attribute
        containing an AnnData object with intensity values for each cell across the channels in the `img_layer`.

    Notes
    -----
    - The function currently supports scenarios where the `img_layer` and `labels_layer` are aligned and have the same
      shape. Misalignments or differences in shape must be handled prior to invoking this function.
    - Intensity calculation is performed per channel for each cell. The function aggregates this information and
      attaches it as a table (AnnData object) within the SpatialData object.
    - Due to the memory-intensive nature of the operation, especially for large datasets, the function implements
      chunk-based processing, aided by Dask. The `chunks` parameter allows for customization of the chunk sizes used
      during processing.
    """

    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Extracting intensities from the last image layer '{img_layer}' of the provided SpatialData object."
        )

    if labels_layer is None:
        labels_layer = [*sdata.labels][-1]
        log.warning(
            f"No labels layer specified. "
            f"Using mask from labels layer '{labels_layer}' of the provided SpatialData object."
        )

    if channels is not None:
        channels = (
            list(channels)
            if isinstance(channels, Iterable) and not isinstance(channels, str)
            else [channels]
        )

    # currently this function will only work if img_layer and labels_layer have the same shape.
    # And are in same position, i.e. if one is translated, other should be translated with same offset
    se_image = _get_spatial_element(sdata, layer=img_layer)
    se_labels = _get_spatial_element(sdata, layer=labels_layer)

    assert (
        se_image.data.shape[1:] == se_labels.data.shape
    ), "Only arrays with same spatial shape are currently supported, "
    f"but image layer with name {img_layer} has shape {se_image.data.shape}, "
    f"while labels layer with name {labels_layer} has shape {se_labels.data.shape}  "

    t1x, t1y = _get_translation(se_image)
    t2x, t2y = _get_translation(se_labels)

    assert (t1x, t1y) == (t2x, t2y), f"image layer with name {img_layer} should "
    f"have same translation as labels layer with name {labels_layer}"

    if channels is None:
        channels = se_image.c.data

    # iterate over all the channels and collect intensity for each channel and each cell
    channel_intensities = []
    for channel in channels:
        channel_idx = list(channels).index(channel)
        channel_intensities.append(
            _calculate_intensity(
                se_image.isel(c=channel_idx).data, se_labels.data, chunks=chunks
            )
        )

    channel_intensities = np.concatenate(channel_intensities, axis=1)

    cells = unique(se_labels.data).compute()
    cells = pd.DataFrame(index=cells)
    cells.index = cells.index.map(str)
    cells.index.name = "cells"

    channels = list(map(str, channels))
    if append_labels_layer_name:
        channels = [f"{channel}_{labels_layer}" for channel in channels]
    var = pd.DataFrame(index=channels)
    var.index = var.index.map(str)
    var.index.name = "channels"

    adata = AnnData(X=channel_intensities, obs=cells, var=var)
    adata.obs["region"] = 1
    adata.obs["instance"] = 1

    if sdata.table is None:
        sdata.table = spatialdata.models.TableModel.parse(
            adata, region_key="region", region=1, instance_key="instance"
        )

        return sdata

    def _sort_str_index(df):
        df.index = df.index.astype(int)  # convert index to integer
        df = df.sort_index()
        df.index = df.index.astype(str)
        return df

    if append:
        df_joined = adata.to_df().join(
            sdata.table.to_df(), how="outer"
        )  # we do not want to throw away cells
        df_joined = _sort_str_index(df_joined)
        # we do not want to loose already computed obs
        # remove column region and instance, because otherwise we can not do a join
        # TODO handle this more carefully if multiple adata will be supported in spatialdata.
        left = adata.obs.drop(columns=["region", "instance"])
        right = sdata.table.obs.drop(columns=["region", "instance"])
        obs_joined = left.join(right, how="outer")
        obs_joined = _sort_str_index(obs_joined)
        adata = AnnData(X=df_joined, obs=obs_joined)
        adata.obs["region"] = 1
        adata.obs["instance"] = 1

    del sdata.table
    sdata.table = spatialdata.models.TableModel.parse(
        adata, region_key="region", region=1, instance_key="instance"
    )

    return sdata


def _calculate_intensity(
    float_dask_array: Array,
    mask_dask_array: Array,
    chunks: Optional[str | int | tuple[int, ...]] = 10000,
) -> NDArray:
    # lazy computation of pixel intensities on one channel for each label in mask_dask_array
    # result is an array of shape (len(unique(mask_dask_array).compute(), 1 ), so be aware that if
    # some labels are missing, e.g. unique(mask_dask_array).compute()=np.array([ 0,1,3,4 ]), resulting
    # array will hold at postiion 2 the intensity for cell with index 3.

    assert float_dask_array.shape == mask_dask_array.shape

    float_dask_array = float_dask_array.rechunk(chunks)
    mask_dask_array = mask_dask_array.rechunk(chunks)

    labels = unique(mask_dask_array).compute()

    def _calculate_intensity_per_chunk(
        mask_block: NDArray, float_block: NDArray
    ) -> NDArray:
        sums = np.bincount(mask_block.ravel(), weights=float_block.ravel())

        num_padding = (max(labels) + 1) - len(sums)

        sums = np.pad(sums, (0, num_padding), "constant", constant_values=(0))

        sums = sums[labels]

        sums = sums.reshape(-1, 1)

        return sums

    chunk_sum = da.map_blocks(
        lambda m, f: _calculate_intensity_per_chunk(m, f),
        mask_dask_array,
        float_dask_array,
        dtype=float,
        chunks=(len(labels), 1),
    )

    # here you could persist array of shape (num_labels * nr_of_chunks_x, nr_of_chunks_y) into memory
    # chunk_sum=chunk_sum.persist()  --> could take a lot of memory if you would have many chunks
    # therefore we use this delayed tasks.

    sum_of_chunks = np.zeros((len(labels), 1), dtype=chunk_sum.dtype)

    num_chunks = chunk_sum.numblocks
    tasks = []

    # sum the result for each chunk
    for i in range(num_chunks[0]):
        for j in range(num_chunks[1]):
            current_chunk = chunk_sum.blocks[i, j]
            task = delayed(np.add)(sum_of_chunks, current_chunk)
            tasks.append(task)

    total_sum_delayed = delayed(sum)(tasks)

    sum_of_chunks = total_sum_delayed.compute()

    return sum_of_chunks
