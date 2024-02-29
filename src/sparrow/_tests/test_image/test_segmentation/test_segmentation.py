import dask.dataframe as dd
import pandas as pd
import spatialdata
from dask.dataframe import DataFrame
from spatialdata import SpatialData

from sparrow.image.segmentation._segmentation import segment, segment_points
from sparrow.image.segmentation.segmentation_models._baysor import _dummy
from sparrow.image.segmentation.segmentation_models._cellpose import _cellpose


def test_segment(sdata_multi_c: SpatialData):
    sdata_multi_c = segment(
        sdata_multi_c,
        img_layer="combine",
        model=_cellpose,
        output_labels_layer="masks_cellpose",
        output_shapes_layer="masks_cellpose_boundaries",
        trim=False,
        chunks=50,
        overwrite=True,
        depth=30,
        crd=[10, 110, 0, 100],
        scale_factors=[2, 2, 2, 2],
        diameter=20,
        cellprob_threshold=-4,
        flow_threshold=0.9,
        model_type="nuclei",
        do_3D=False,
        channels=[1, 0],
    )

    assert "masks_cellpose" in sdata_multi_c.labels
    assert "masks_cellpose_boundaries" in sdata_multi_c.shapes
    assert isinstance(sdata_multi_c, SpatialData)


def test_segment_3D(sdata_multi_c: SpatialData):
    sdata_multi_c = segment(
        sdata_multi_c,
        img_layer="combine_z",
        model=_cellpose,
        output_labels_layer="masks_cellpose_3D",
        output_shapes_layer="masks_cellpose_3D_boundaries",
        trim=False,
        chunks=(50, 50),
        overwrite=True,
        depth=(20, 20),
        crd=[50, 80, 10, 70],
        scale_factors=[2],
        diameter=20,
        cellprob_threshold=-4,
        flow_threshold=0.9,
        model_type="nuclei",
        channels=[1, 0],
        do_3D=True,
        anisotropy=1,
    )

    assert "masks_cellpose_3D" in sdata_multi_c.labels
    assert isinstance(sdata_multi_c, SpatialData)


def test_segment_points(sdata_multi_c: SpatialData):
    data = {"x": [10], "y": [10], "gene": ["dummy_gene"]}

    # Create the DataFrame
    df = pd.DataFrame(data)

    ddf = dd.from_pandas(df, npartitions=1)

    coordinates = {"x": "x", "y": "y"}

    sdata_multi_c.add_points(
        name="transcripts",
        points=spatialdata.models.PointsModel.parse(
            ddf,
            coordinates=coordinates,
        ),
    )
    assert type(sdata_multi_c.points["transcripts"]) == DataFrame

    sdata_multi_c = segment_points(
        sdata_multi_c,
        labels_layer="masks_whole",
        points_layer="transcripts",
        name_x="x",
        name_y="y",
        name_gene="gene",
        model=_dummy,
        output_labels_layer="masks_whole_copy_dummy",
        output_shapes_layer="masks_whole_copy_dummy_boundaries",
        chunks=256,
        crd=None,
    )
