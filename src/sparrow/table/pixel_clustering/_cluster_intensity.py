from __future__ import annotations

from pathlib import Path
from typing import Iterable

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element
from sparrow.table._allocation_intensity import allocate_intensity
from sparrow.table._preprocess import preprocess_proteomics
from sparrow.table._table import ProcessTable, _add_table_layer
from sparrow.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _METACLUSTERING_KEY


def cluster_intensity(
    sdata: SpatialData,
    mapping: pd.Series,  # pandas series with at the index the clusters and as values the metaclusters
    img_layer: str,
    labels_layer: str,
    output_layer: str,
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    chunks: str | int | tuple[int, ...] | None = 10000,
    overwrite=False,
) -> SpatialData:
    """
    Calculates average intensity of each channel in `img_layer` per SOM cluster as available in the `labels_layer`, and saves it as a table layer in `sdata` as `output_layer`.

    This function computes average intensity for each SOM cluster identified in the `labels_layer` and stores the results in a new table layer.
    The intensity calculation can be subset by channels and adjusted for chunk size for efficient processing. SOM clusters can be calculated using `sp.im.flowsom`.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    mapping : pd.Series
        A pandas Series mapping SOM cluster IDs (index) to metacluster IDs (values).
    img_layer : str
        The image layer of `sdata` from which the intensity is calculated.
    labels_layer : str
        The labels layer in `sdata` that contains the SOM cluster IDs. I.e. the `output_layer_clusters` labels layer obtained through `sp.im.flowsom`.
    output_layer : str
        The output table layer in `sdata` where results are stored.
    channels : int | str | Iterable[int] | Iterable[str] | None, optional
        Specifies the channels to be included in the intensity calculation.
    chunks : str | int | tuple[int, ...] | None, optional
        Chunk sizes for processing. If provided as a tuple, it should contain chunk sizes for `c`, `(z)`, `y`, `x`.
    overwrite : bool, default=False
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    SpatialData
        The input `sdata` with the new table layer added.

    Warnings
    --------
    - Ensure that all SOM cluster IDs in `labels_layer` exist within the provided mapping Series; otherwise, an assertion error will occur.
    - The function is designed for use with spatial proteomics data and assumes the input data is appropriately preprocessed.

    Raises
    ------
    AssertionError
        If some labels in `labels_layer` are not found in the provided mapping pandas Series.
    """
    se = _get_spatial_element(sdata, layer=labels_layer)

    labels = da.unique(se.data).compute()

    assert np.all(
        np.in1d(labels[labels != 0], mapping.index.astype(int))
    ), "Some labels in `labels_layer` could not be found in the provided pandas Series that maps SOM cluster ID's to metacluster IDs."

    # allocate the intensity to the clusters
    sdata = allocate_intensity(
        sdata,
        img_layer=img_layer,
        labels_layer=labels_layer,
        output_layer=output_layer,
        channels=channels,
        chunks=chunks,
        remove_background_intensity=True,
        append=False,
        overwrite=overwrite,
    )

    # for size normalization of cluster intensities
    sdata = preprocess_proteomics(
        sdata,
        labels_layer=labels_layer,
        table_layer=output_layer,
        output_layer=output_layer,
        size_norm=True,
        log1p=False,
        scale=False,
        calculate_pca=False,
        overwrite=True,
    )

    # append metacluster labels to the table using the mapping
    mapping = mapping.reset_index().rename(columns={"index": _INSTANCE_KEY})  # _INSTANCE_KEY is the cluster ID
    mapping[_INSTANCE_KEY] = mapping[_INSTANCE_KEY].astype(int)
    # get the adata
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=output_layer)
    adata = process_table_instance._get_adata()
    old_index = adata.obs.index
    adata.obs = pd.merge(adata.obs.reset_index(), mapping, on=[_INSTANCE_KEY], how="left")
    adata.obs.index = old_index
    adata.obs = adata.obs.drop(columns=[_CELL_INDEX])

    assert not adata.obs[_METACLUSTERING_KEY].isna().any(), "Not all SOM cluster IDs could be linked to a metacluster."

    # calculate mean intensity per metacluster
    df = adata.to_df().copy()
    df[[_CELLSIZE_KEY, _METACLUSTERING_KEY]] = adata.obs[[_CELLSIZE_KEY, _METACLUSTERING_KEY]].copy()

    if channels is None:
        channels = adata.var.index.values
    else:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]

    df = _mean_intensity_per_metacluster(df, channels=channels)

    adata.uns[f"{_METACLUSTERING_KEY}"] = df

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=True,
    )

    return sdata


def _mean_intensity_per_metacluster(df, channels: Iterable[str]):
    # Assuming df is your dataframe
    def weighted_mean(x, data, weight):
        """Calculate weighted mean for a column."""
        return (x * data[weight]).sum() / data[weight].sum()

    # Calculate weighted average for each marker per pixel_meta_cluster
    weighted_averages = df.groupby(
        _METACLUSTERING_KEY,
    ).apply(
        lambda x: pd.Series(
            {col: weighted_mean(x[col], x, _CELLSIZE_KEY) for col in channels},
        ),
        include_groups=False,
    )

    return weighted_averages.reset_index()


def _export_to_ark_format(adata: AnnData, output: str | Path | None) -> pd.DataFrame:
    """Export avg intensity per SOM cluster calculated via `sp.tb.cluster_intensity` to a csv file that can be visualized by the ark gui."""
    df = adata.to_df().copy()
    df["pixel_meta_cluster"] = adata.obs[_METACLUSTERING_KEY].copy()
    df["pixel_som_cluster"] = adata.obs[_INSTANCE_KEY].copy()
    df["count"] = adata.obs[_CELLSIZE_KEY].copy()

    if output is not None:
        df.to_csv(output, index=False)

    return df


def _grouped_obs_mean(adata: AnnData, group_key: str, layer: str = None) -> pd.DataFrame:
    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X

    grouped = adata.obs.groupby(group_key)
    columns = list(grouped.groups.keys())
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64), columns=columns, index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out
