from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from shapely.geometry import LineString, MultiLineString
from spatialdata.models import get_axes_names
from xarray import DataArray


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


def _get_raster_multiscale(element: MultiscaleSpatialImage) -> list[DataArray]:
    if not isinstance(element, MultiscaleSpatialImage):
        raise TypeError(f"Unsupported type for images or labels: {type(element)}")

    axes = get_axes_names(element)

    if "c" in axes:
        assert axes.index("c") == 0

    # sanity check
    scale_0 = element.__iter__().__next__()
    v = element[scale_0].values()
    assert len(v) == 1

    list_of_xdata = []
    for k in element:
        v = element[k].values()
        assert len(v) == 1
        xdata = v.__iter__().__next__()
        list_of_xdata.append(xdata)

    return list_of_xdata


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
