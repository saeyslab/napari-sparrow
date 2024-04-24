from pathlib import Path

import flowsom as fs
import numpy as np
import pandas as pd
from anndata import AnnData
from pandas import DataFrame


def _export_to_ark_format(adata: AnnData, fsom: fs.FlowSOM, output: str | Path) -> DataFrame:
    # add average intensity per SOM cluster to adata
    df_grouped = _grouped_obs_mean(adata, group_key="clustering")
    df_grouped = df_grouped.transpose().reset_index(names="clustering")

    assert (
        fsom.get_cluster_data().obs.index.astype(int) == df_grouped["clustering"]
    ).all(), "'fsom.get_cluster_data()' should be sorted as 'adata.obs.clustering'"

    # probably need to sort before adding to table
    df_grouped["count"] = (fsom.get_cluster_data().obs["percentages"] * adata.shape[0]).values.astype(int)
    df_grouped["metaclustering"] = fsom.get_cluster_data().obs["metaclustering"].values.astype(int)

    df_grouped.rename(columns={"clustering": "pixel_som_cluster", "metaclustering": "pixel_meta_cluster"}, inplace=True)

    # ark wants to have pixel_som_cluster indices starting from 1
    df_grouped["pixel_som_cluster"] = df_grouped["pixel_som_cluster"] + 1

    df_grouped.to_csv(output, index=False)

    return df_grouped


def _grouped_obs_mean(adata: AnnData, group_key: str, layer: str = None) -> DataFrame:
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
