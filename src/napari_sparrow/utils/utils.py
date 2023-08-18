import os
from pathlib import Path
from typing import Any, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from anndata import AnnData
from geopandas import GeoDataFrame
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from shapely.geometry import LineString, MultiLineString


def parse_subset(subset):
    """
    e.g $ sparrow subset=\'0:100,0:100\'
    >>> parse_subset('0:100,0:100')
    return left corner and size ([0, 0], [100, 100])

    """
    left_corner = []
    size = []
    for x in subset.split(","):
        left_corner.append(int(x.split(":")[0]))
        size.append(int(x.split(":")[1]) - left_corner[-1])
    return left_corner, size


def ic_to_da(
    ic, label="image", drop_dims=["z", "channels"], reduce_z=None, reduce_c=None
):
    """
    Convert ImageContainer to dask array.
    ImageContainer defaults to (x, y, z, channels (c if using xarray format)), most of the time we need just (x, y)
    The c channel will be named c:0 after segmentation.
    """
    if reduce_z or reduce_c:
        # TODO solve c:0 output when doing .isel(z=reduce_z, c=reduce_c)
        return ic[label].isel({"z": reduce_z, "c:0": 0}).data
    else:
        return ic[label].squeeze(dim=drop_dims).data


def linestring_to_arrays(geometries):
    arrays = []
    for geometry in geometries:
        if isinstance(geometry, LineString):
            arrays.extend(list(geometry.coords))
        elif isinstance(geometry, MultiLineString):
            for item in geometry.geoms:
                arrays.extend(list(item.coords))
    return np.array(arrays)


# https://github.com/scverse/napari-spatialdata/blob/main/src/napari_spatialdata/_viewer.py#L105
def _get_polygons_in_napari_format(df: GeoDataFrame) -> List:
    polygons = []
    # affine = _get_transform(sdata.shapes[key], selected_cs)

    # when mulitpolygons are present, we select the largest ones
    if "MultiPolygon" in np.unique(df.geometry.type):
        # logger.info("Multipolygons are present in the data. Only the largest polygon per cell is retained.")
        df = df.explode(index_parts=False)
        df["area"] = df.area
        df = df.sort_values(by="area", ascending=False)  # sort by area
        df = df[~df.index.duplicated(keep="first")]  # only keep the largest area
        df.index = df.index.astype(int)  # convert index to integer
        df = df.sort_index()
        df.index = df.index.astype(str)

    if len(df) < 100:
        for i in range(0, len(df)):
            polygons.append(list(df.geometry.iloc[i].exterior.coords))
    else:
        for i in range(
            0, len(df)
        ):  # This can be removed once napari is sped up in the plotting. It changes the shapes only very slightly
            polygons.append(
                list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords)
            )
    # this will only work for polygons and not for multipolygons
    # switch x,y positions of polygon indices, napari wants (y,x)
    polygons = _swap_coordinates(polygons)

    return polygons


def _swap_coordinates(data: list[Any]) -> list[Any]:
    return [[(y, x) for x, y in sublist] for sublist in data]


##TODO:rewrite this function
def extract(ic: sq.im.ImageContainer, adata: AnnData) -> AnnData:
    """This function performs segmentation feature extraction and adds cell area and mean intensity to the annData object under obsm segmentation_features."""
    sq.im.calculate_image_features(
        adata,
        ic,
        layer="raw",
        features="segmentation",
        key_added="segmentation_features",
        features_kwargs={
            "segmentation": {
                "label_layer": "segment_cellpose",
                "props": ["label", "area", "mean_intensity"],
                "channels": [0],
            }
        },
    )

    return adata


def color(_) -> matplotlib.colors.Colormap:
    """Select random color from set1 colors."""
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    """Select border color from tab10 colors or preset color (1, 1, 1, 1) otherwise."""
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    """Select linewidth 1 if true else 0.5."""
    return 1 if r else 0.5


def _export_config(cfg: DictConfig, output_yaml: str | Path):
    yaml_config = OmegaConf.to_yaml(cfg)
    output_dir = os.path.dirname(output_yaml)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_yaml, "w") as f:
        f.write(yaml_config)
