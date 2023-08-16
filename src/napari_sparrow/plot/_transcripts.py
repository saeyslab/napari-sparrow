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
):
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        crd=crd,
        figsize=figsize,
        output=output,
    )
