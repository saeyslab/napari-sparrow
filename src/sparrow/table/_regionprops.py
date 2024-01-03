from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.measure import moments, regionprops, regionprops_table
from skimage.measure._regionprops import RegionProperties
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element
from sparrow.table._table import _back_sdata_table_to_zarr
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def add_regionprop_features(
    sdata: SpatialData,
    labels_layer: Optional[str] = None,
    append_labels_layer_name: bool = True,
):
    """
    Enhances a SpatialData object with region property features calculated from the specified labels layer,
    updating its table attribute with these computed cellular properties.

    This function computes various geometric and morphological properties for each labeled region (presumed cells)
    found in the specified layer of the SpatialData object. These properties include measures such as area,
    eccentricity, axis lengths, perimeter, and several custom ratios and metrics providing insights into
    each cell's shape and structure. The calculated properties are appended to the observations in the SpatialData
    object's underlying table.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing spatial information about cells/nuclei. This object will be updated with
        new region property features.
    labels_layer : str, optional
        The name of the layer in `sdata` that contains the labeled regions, typically derived from a segmentation
        process. Each distinct label corresponds to a different cell, and properties will be calculated for these
        labeled regions. If not provided, the function will infer or require a default layer.
    append_labels_layer_name: bool, optional.
        If set to True, labels_layer will be added as suffix to the name of the region property features in the sdata.table.obs
        Dataframe.

    Returns
    -------
    SpatialData
        The original SpatialData object, updated to include a range of new region-specific property measurements
        in its `sdata.table.obs` attribute.

    Notes
    -----
    - The function operates by pulling the required data (masks) into memory for processing, as the underlying 'regionprops'
      functionality does not support lazy loading. Consequently, sufficient memory must be available for large datasets.
    - Computed properties are joined with the existing observations within the SpatialData's table, expanding the
      dataset's feature set.
    """

    if labels_layer is None:
        labels_layer = [*sdata.labels][-1]
        log.warning(
            f"No labels layer specified. "
            f"Using mask from labels layer '{labels_layer}' of the provided SpatialData object."
        )

    se = _get_spatial_element(sdata, layer=labels_layer)

    # pull masks in in memory. skimage.measure.regionprops does not work with lazy objects.
    masks = se.data.compute()

    cell_props = _calculate_regionprop_features(masks)

    if append_labels_layer_name:
        new_columns = [f"{col}_{labels_layer}" for col in cell_props.columns]
        cell_props.columns = new_columns

    # keep all cells in sdata.table, but append properties to .obs  # left or outer??
    # add a warning if cell_props contains more cells than sdata.table.obs

    extra_cells = cell_props.index.difference(sdata.table.obs.index)
    if not extra_cells.empty:
        log.warning(
            f"Calculated properties of { len( extra_cells ) } cells/nuclei obtained from labels layer '{labels_layer}' "
            "will not be added to 'sdata.table.obs', because their indicices are not in 'sdata.table'. "
            "Please first append their intensities with the 'allocate_intensity' function "
            "if they should be included."
        )

    sdata.table.obs = sdata.table.obs.join(cell_props, how="left")

    _back_sdata_table_to_zarr(sdata=sdata)

    return sdata


def _calculate_regionprop_features(
    masks: NDArray,
) -> DataFrame:
    properties = [
        "label",
        "area",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "centroid",
        "convex_area",
        "equivalent_diameter",
    ]

    cell_props = DataFrame(regionprops_table(masks, properties=properties))

    props = regionprops(masks)

    for _func in [
        _major_minor_axis_ratio,
        _perim_square_over_area,
        _major_axis_equiv_diam_ratio,
        _convex_hull_resid,
        _centroid_dif,
    ]:
        results = []
        for prop in props:
            results.append(_func(prop))
        cell_props[_func.__name__] = results

    cell_props = cell_props.set_index("label")
    cell_props.index.name = "cells"
    cell_props.index = cell_props.index.map(str)

    return cell_props


# following helper functions taken from
# https://github.com/angelolab/ark-analysis/blob/main/src/ark/segmentation/regionprops_extraction.py


def _major_minor_axis_ratio(prop: RegionProperties) -> float:
    if prop.minor_axis_length == 0:
        return np.float("NaN")
    else:
        return prop.major_axis_length / prop.minor_axis_length


def _perim_square_over_area(prop: RegionProperties) -> float:
    return np.square(prop.perimeter) / prop.area


def _major_axis_equiv_diam_ratio(prop: RegionProperties) -> float:
    return prop.major_axis_length / prop.equivalent_diameter


def _convex_hull_resid(prop: RegionProperties) -> float:
    return (prop.convex_area - prop.area) / prop.convex_area


def _centroid_dif(prop: RegionProperties) -> float:
    cell_image = prop.image
    cell_M = moments(cell_image)
    cell_centroid = np.array([cell_M[1, 0] / cell_M[0, 0], cell_M[0, 1] / cell_M[0, 0]])

    convex_image = prop.convex_image
    convex_M = moments(convex_image)
    convex_centroid = np.array(
        [convex_M[1, 0] / convex_M[0, 0], convex_M[0, 1] / convex_M[0, 0]]
    )

    centroid_dist = np.linalg.norm(cell_centroid - convex_centroid) / np.sqrt(prop.area)

    return centroid_dist
