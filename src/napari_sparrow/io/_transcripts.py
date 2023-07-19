from pathlib import Path
from typing import Optional, Union
import numpy as np
import dask.dataframe as dd
from dask.dataframe.core import DataFrame as DaskDataFrame
import spatialdata
from spatialdata import SpatialData


def read_resolve_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
) -> SpatialData:
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 0,
        "column_y": 1,
        "column_gene": 3,
        "delimiter": "\t",
        "header": None,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_vizgen_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    path_transform_matrix: str | Path,
) -> SpatialData:
    args = (sdata, path_count_matrix, path_transform_matrix)
    kwargs = {
        "column_x": 2,
        "column_y": 3,
        "column_gene": 8,
        "delimiter": ",",
        "header": 0,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_stereoseq_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
) -> SpatialData:
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 1,
        "column_y": 2,
        "column_gene": 0,
        "column_midcount": 3,
        "delimiter": ",",
        "header": 0,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_transcripts(
    sdata: SpatialData,
    path_count_matrix: Union[str, Path],
    path_transform_matrix: Optional[Union[str, Path]] = None,
    debug: bool = False,
    column_x: int = 0,
    column_y: int = 1,
    column_gene: int = 3,
    column_midcount: Optional[int] = None,
    delimiter: str = ",",
    header: Optional[int] = None,
) -> SpatialData:
    # Read the CSV file using Dask
    ddf = dd.read_csv(path_count_matrix, delimiter=delimiter, header=header)

    # Read the transformation matrix
    if path_transform_matrix is None:
        print("No transform matrix given, will use identity matrix.")
        transform_matrix = np.identity(3)
    else:
        transform_matrix = np.loadtxt(path_transform_matrix)

    print(transform_matrix)

    if debug:
        n = 100000
        fraction = n / len(ddf)
        ddf = ddf.sample(frac=fraction)

    # Function to repeat rows based on MIDCount value
    def repeat_rows(df):
        repeat_df = df.reindex(
            df.index.repeat(df.iloc[:, column_midcount])
        ).reset_index(drop=True)
        return repeat_df

    # Apply the row repeat function if column_midcount is not None (e.g. for Stereoseq)
    if column_midcount is not None:
        ddf = ddf.map_partitions(repeat_rows, meta=ddf)

    def transform_coordinates(df):
        micron_coordinates = df.iloc[:, [column_x, column_y]].values
        micron_coordinates = np.column_stack(
            (micron_coordinates, np.ones(len(micron_coordinates)))
        )
        pixel_coordinates = np.dot(micron_coordinates, transform_matrix.T)[:, :2]
        result_df = df.iloc[:, [column_gene]].copy()
        result_df["pixel_x"] = pixel_coordinates[:, 0]
        result_df["pixel_y"] = pixel_coordinates[:, 1]
        return result_df

    # Apply the transformation to the Dask DataFrame
    transformed_ddf = ddf.map_partitions(transform_coordinates)

    # Rename the columns
    transformed_ddf.columns = ["gene", "pixel_x", "pixel_y"]

    # Reorder
    transformed_ddf = transformed_ddf[["pixel_x", "pixel_y", "gene"]]

    sdata = _add_transcripts_to_sdata(sdata, transformed_ddf)

    return sdata


def _add_transcripts_to_sdata(sdata: SpatialData, transformed_ddf: DaskDataFrame):
    # TODO below fix to remove transcripts does not work when backed by zarr store, points not allowed to be deleted on disk.
    if sdata.points:
        for points_layer in [*sdata.points]:
            del sdata.points[points_layer]

    sdata.add_points(
        name="transcripts",
        points=spatialdata.models.PointsModel.parse(
            transformed_ddf, coordinates={"x": "pixel_x", "y": "pixel_y"}
        ),
    )
    return sdata
