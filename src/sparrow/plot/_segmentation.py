from __future__ import annotations

from typing import Any, Dict, Optional,Tuple, List
from pathlib import Path

from spatialdata import SpatialData

from sparrow.plot import plot_shapes


def segment(
    sdata: SpatialData,
    img_layer: str = "raw_image",
    shapes_layer: str = "segmentation_mask_boundaries",
    channel: Optional[ int | List[ int] ] = None,
    crd:Optional[Tuple[int,int,int,int]]=None,
    output:Optional[ str | Path ]=None,
    **kwargs: Dict[str, Any],
):
    """
    Visualize obtained shapes layer (i.e. segmentation mask boundaries) from a SpatialData object.

    This function utilizes the `plot_shapes` method to display the segmentation results from the provided SpatialData object.
    Final plot will contain tow subplots, left the image without provided shapes layer overlay, and the right subplot with
    shapes layer overlay.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    img_layer : str, optional
        Name of the image layer to be visualized, by default "raw_image".
    shapes_layer : str, optional
        Name of the layer containing segmentation mask boundaries, by default "segmentation_mask_boundaries".
    channel : int or list of int, optional
        The channel(s) of the image to be visualized. If None, all channels are considered, by default None.
    crd : tuple of int, optional
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    output : str or Path, optional
        Path to save the output image. If None, the image will not be saved and will be displayed instead, by default None.
    **kwargs : dict
        Additional keyword arguments to be passed to the `plot_shapes` function.

    Returns
    -------
    None

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> segment(sdata, img_layer="raw_img", crd=(2000,4000,2000,4000))
    """
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=[None, shapes_layer],
        channel=channel,
        crd=crd,
        output=output,
        **kwargs,
    )