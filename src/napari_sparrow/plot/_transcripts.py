from typing import Optional, Tuple
from pathlib import Path

from spatialdata import SpatialData

from napari_sparrow.plot import plot_shapes

def transcript_density(
    sdata: SpatialData,
    img_layer: Tuple[ str, str ] =[ "raw_image", "transcript_density" ],
    channel: int = 0,
    crd:Optional[Tuple[int,int,int,int]]=None,
    figsize: Optional[ Tuple[int,int ] ]=None,
    output:Optional[ str | Path ]=None,
)->None:
    """
    Visualize the transcript density layer.

    This function wraps around the `plot_shapes` function to showcase transcript density.

    Parameters:
    ----------
    sdata: SpatialData
        Data containing spatial information for plotting.
    img_layer: Tuple[str, str], default=["raw_image", "transcript_density"]
        A tuple where the first element indicates the base image layer and 
        the second element indicates the transcript density.
    channel: int, default=0
        The channel of the image to be visualized. 
        If the channel not in one of the images, the first available channel of the image will be plotted
    crd: Optional[Tuple[int, int, int, int]], default=None
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    figsize: Optional[Tuple[int, int]], default=None
        The figure size for the visualization. If None, a default size will be used.
    output: Optional[str | Path], default=None
        Path to save the output image. If None, the image will not be saved and will be displayed instead.

    Returns:
    -------
    None

    Example:
    -------
    >>> sdata = SpatialData(...)
    >>> transcript_density(sdata, img_layer=["raw_img", "density"], crd=(2000,4000,2000,4000))
    """

    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        crd=crd,
        figsize=figsize,
        output=output,
    )