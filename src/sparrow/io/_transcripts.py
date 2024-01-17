from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import dask.dataframe as dd
import numpy as np
import spatialdata
from dask.dataframe.core import DataFrame as DaskDataFrame
from spatialdata import SpatialData

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def read_resolve_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    overwrite: bool = False,
) -> SpatialData:
    """
    Reads and adds Resolve transcript information to a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to which the transcripts will be added.
    path_count_matrix : str | Path
        Path to the file containing the transcripts information specific to Resolve.
        Expected to contain x, y coordinates and a gene name.
    overwrite: bool, default=False
        If True overwrites the element (points layer) if it already exists.

    Returns
    -------
    SpatialData
        The updated SpatialData object containing the transcripts.
    """
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 0,
        "column_y": 1,
        "column_gene": 3,
        "delimiter": "\t",
        "header": None,
        "overwrite": overwrite,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_vizgen_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    path_transform_matrix: str | Path,
    overwrite: bool = False,
) -> SpatialData:
    """
    Reads and adds Vizgen transcript information to a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to which the transcripts will be added.
    path_count_matrix : str | Path
        Path to the file containing the transcripts information specific to Vizgen.
        Expected to contain x, y coordinates and a gene name.
    path_transform_matrix : str | Path
        Path to the transformation matrix for the affine transformation.
    overwrite: bool, default=False
        If True overwrites the element (points layer) if it already exists.

    Returns
    -------
    SpatialData
        The updated SpatialData object containing the transcripts.
    """
    args = (sdata, path_count_matrix, path_transform_matrix)
    kwargs = {
        "column_x": 2,
        "column_y": 3,
        "column_z": 4,
        "column_gene": 8,
        "delimiter": ",",
        "header": 0,
        "overwrite": overwrite,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_stereoseq_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    overwrite: bool = False,
) -> SpatialData:
    """
    Reads and adds Stereoseq transcript information to a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to which the transcripts will be added.
    path_count_matrix : str | Path
        Path to the file containing the transcripts information specific to Stereoseq.
        Expected to contain x, y coordinates, gene name, and a midcount column.
    overwrite: bool, default=False
        If True overwrites the element (points layer) if it already exists.

    Returns
    -------
    SpatialData
        The updated SpatialData object containing the transcripts.
    """
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 1,
        "column_y": 2,
        "column_gene": 0,
        "column_midcount": 3,
        "delimiter": ",",
        "header": 0,
        "overwrite": overwrite,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_transcripts(
    sdata: SpatialData,
    path_count_matrix: Union[str, Path],
    path_transform_matrix: Optional[Union[str, Path]] = None,
    points_layer: str = "transcripts",
    overwrite: bool = False,
    debug: bool = False,
    column_x: int = 0,
    column_y: int = 1,
    column_z: Optional[int] = None,
    column_gene: int = 3,
    column_midcount: Optional[int] = None,
    delimiter: str = ",",
    header: Optional[int] = None,
    comment: Optional[str] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    filter_gene_names: Optional(str|list)=None 
) -> SpatialData:
    """
    Reads transcript information from a file with each row listing the x and y coordinates, along with the gene name.
    If a transform matrix is provided a linear transformation is applied to the coordinates of the transcripts.
    The SpatialData object is augmented with a points layer named 'transcripts' that contains the transcripts.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object to which the transcripts will be added.
    path_count_matrix : Union[str, Path]
        Path to the .txt file containing the transcripts information. Each row should contain an x, y coordinate and a gene name.
        Optional a midcount column is provided. If a midcount column is provided, rows are repeated.
    path_transform_matrix : Optional[Union[str, Path]], default=None
        This file should contain a 3x3 transformation matrix for the affine transformation.
        The matrix defines the linear transformation to be applied to the coordinates of the transcripts.
        If no transform matrix is specified, the identity matrix will be used.
    points_layer: str, default='transcripts'.
        Name of the points layer of the SpatialData object to which the transcripts will be added.
    overwrite: bool, default=False
        If True overwrites the element (points layer) if it already exists.
    debug : bool, default=False
        If True, a sample of the data is processed for debugging purposes.
    column_x : int, default=0
        Column index of the X coordinate in the count matrix.
    column_y : int, default=1
        Column index of the Y coordinate in the count matrix.
    column_z : int, default=None
        Column index of the Z coordinate in the count matrix.
    column_gene : int, default=3
        Column index of the gene information in the count matrix.
    column_midcount : Optional[int], default=None
        Column index for the count value to repeat rows in the count matrix. Ignored when set to None.
    delimiter : str, default=","
        Delimiter used to separate values in the CSV file.
    header : Optional[int], default=None
        Row number to use as the header in the CSV file. If None, no header is used.
    comment : Optional[str], default=None
        Character indicating that the remainder of line should not be parsed.
        If found at the beginning of a line, the line will be ignored altogether.
        This parameter must be a single character.
    crd : tuple of int, optional
        The coordinates (in pixels) for the region of interest in the format (xmin, xmax, ymin, ymax).
        If None, all transcripts are considered.
    filter_gene_names: list or list of strings, optional
        Regular expression(s) of gene names that need to be filtered out, mostly control genes that were added, and which you don't want to use. If list of strings, all items in the list are seen as regular expressions.  
    Returns
    -------
    SpatialData
        The updated SpatialData object containing the transcripts.

    Notes
    -----
    This function reads the CSV file using Dask and applies a transformation matrix to the coordinates.
    It can also repeat rows based on the MIDCount value and can work in a debug mode that samples the data.
    """
    # Read the CSV file using Dask
    ddf = dd.read_csv(path_count_matrix, delimiter=delimiter, header=header, comment=comment)
    
    def filter_names(ddf,column_gene,filter_name):
        # filter out control genes that you don't want ending up in the dataset
        
        ddf=ddf[ddf.iloc[:,column_gene].str.contains(filter_name)==False]
        return ddf
    
    if filter_gene_names:
        
        if type(filter_gene_names) is list:
            
            for i in filter_gene_names:
                ddf=filter_names(ddf,column_gene,i)
                
        elif isinstance(filter_gene_names,str):
            
            ddf=filter_names(ddf,column_gene,filter_gene_names)
        else: 
            log.info('instance to filter on isn\'t a string nor a list. No genes are filtered out based on the gene name. ')
            
    # Read the transformation matrix
    if path_transform_matrix is None:
        log.info("No transform matrix given, will use identity matrix.")
        transform_matrix = np.identity(3)
    else:
        transform_matrix = np.loadtxt(path_transform_matrix)

    log.info(f"Transform matrix used:\n {transform_matrix}")
    
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

    columns = ["pixel_x", "pixel_y", "gene"]
    coordinates = {"x": "pixel_x", "y": "pixel_y"}

    if column_z is not None:
        transformed_ddf["pixel_z"] = ddf.iloc[:, column_z]
        columns.append("pixel_z")
        coordinates["z"] = "pixel_z"

    # Reorder
    transformed_ddf = transformed_ddf[columns]

    if crd is not None:
        transformed_ddf = transformed_ddf.query(
            f"{crd[0]} <= pixel_x < {crd[1]} and {crd[2]} <= pixel_y < {crd[3]}"
        )

    if sdata.points:
        for points_layer in [*sdata.points]:
            del sdata.points[points_layer]

    sdata = _add_transcripts_to_sdata(
        sdata,
        transformed_ddf=transformed_ddf,
        points_layer=points_layer,
        coordinates=coordinates,
        overwrite=overwrite,
    )

    return sdata


def _add_transcripts_to_sdata(
    sdata: SpatialData,
    transformed_ddf: DaskDataFrame,
    points_layer: str,
    coordinates: Dict[str, str],
    overwrite: bool = False,
):

    sdata.add_points(
        name=points_layer,
        points=spatialdata.models.PointsModel.parse(
            transformed_ddf,
            coordinates=coordinates,
        ),
        overwrite=overwrite,
    )
    return sdata
