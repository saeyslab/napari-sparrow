from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from spatialdata import SpatialData

from napari_sparrow.plot import plot_shapes


def tiling_correction(
    sdata: SpatialData,
    img_layer: Tuple[str, str] = ["raw_image", "tiling_correction"],
    channel: Optional[int | Iterable[int]] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    img_title=True,
    channel_title=True,
    output: Optional[str | Path] = None,
) -> None:
    """
    Visualizes the effect of tiling correction.

    This function plots the raw image layer of a SpatialData object alongside the tiling corrected version, allowing for a visual
    comparison between the two. This can be useful for assessing the effectiveness of tiling correction methods
    applied to an image layer of a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    img_layer : Tuple[str, str], optional
        Tuple where the first string represents the layer name for the raw image and the second string
        represents the layer name for the tiling corrected image. Default is ["raw_image", "tiling_correction"].
        Images will be plotted next to each other.
    channel : int or Iterable[int], optional
        Specifies the channel or channels to visualize. If not provided, all channels are used.
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
        If provided, only this region is visualized, by default None.
    figsize : Tuple[int, int], optional
        Size of the generated figure for visualization.
    img_title : bool, optional
        Whether to display the image title on the visualization, by default True.
    channel_title : bool, optional
        Whether to display the channel title on the visualization, by default True.
    output : str or Path, optional
        Path where the generated visualization will be saved. If not provided, the visualization
        is only displayed and not saved.

    Returns
    -------
    None

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> tiling_correction(sdata, img_layer=["original", "corrected"], crd=(2000, 4000, 2000, 4000))

    """
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        crd=crd,
        figsize=figsize,
        img_title=img_title,
        channel_title=channel_title,
        output=output,
    )


def flatfield(flatfield: np.ndarray, output: Optional[str | Path] = None) -> None:
    """
    Visualize the correction performed per tile using a flatfield image.

    This function generates and displays a visualization of the flatfield image, which represents
    correction performed per tile. It can optionally save the generated image to a specified path.

    Parameters
    ----------
    flatfield : np.ndarray
        A 2D numpy array representing the flatfield image used for correction.
    output : str or Path, optional
        Path where the generated visualization will be saved. If not provided, the visualization
        is displayed but not saved. Default is None.

    Returns
    -------
    None

    Examples
    --------
    >>> img = np.array(...)
    >>> flatfield(img, output="path/to/save/image.png")
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(flatfield, cmap="gray")
    ax.set_title("Correction performed per tile")

    plt.tight_layout()
    # Save the plot to output
    if output:
        fig.savefig(output)
    else:
        plt.show()
    plt.close()
