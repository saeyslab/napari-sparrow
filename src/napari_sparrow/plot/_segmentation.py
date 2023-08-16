from typing import Any, Dict, Optional,Tuple, List
from pathlib import Path

from spatialdata import SpatialData

from napari_sparrow.plot import plot_shapes


def segment(
    sdata: SpatialData,
    img_layer: str = "raw_image",
    shapes_layer: str = "segmentation_mask_boundaries",
    channel: Optional[ int | List[ int] ] = None,
    crd:Optional[Tuple[int,int,int,int]]=None,
    output:Optional[ str | Path ]=None,
    **kwargs: Dict[str, Any],
):
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=[None, shapes_layer],
        channel=channel,
        crd=crd,
        output=output,
        **kwargs,
    )