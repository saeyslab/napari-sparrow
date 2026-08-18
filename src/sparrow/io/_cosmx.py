from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import Affine, Identity, Translation, get_transformation, set_transformation
from spatialdata_io import cosmx as sdata_cosmx
from spatialdata_io._constants._constants import CosmxKeys

from sparrow.utils._keys import _GENES_KEY, _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def cosmx(
    path: str | Path | list[str] | list[Path],
    to_coordinate_system: str | list[str] = "global",
    dataset_id: str | list[str] | None = None,
    transcripts: bool = True,
    keep_gene_names: str | Path | Iterable[str] | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    output: str | Path | None = None,
) -> SpatialData:
    """Read *CosMx Nanostring* data into a ``SpatialData`` object.

    This wrapper delegates file parsing and FOV-specific coordinate transforms to
    :func:`spatialdata_io.cosmx`, while adding support for multiple datasets,
    Sparrow coordinate-system naming, optional Zarr backing, and filtering by a
    list of genes to keep.

    Parameters
    ----------
    path
        Path to the CosMx dataset root containing the counts, metadata,
        transcript, image, and label files. A list combines multiple datasets.
    to_coordinate_system
        Coordinate system assigned to each dataset. If a list is provided, its
        length must match ``path`` and every coordinate system must be unique.
    dataset_id
        Dataset identifier used by ``spatialdata_io.cosmx``. A scalar identifier
        is reused for every path; a list must match the length of ``path``.
    transcripts
        Whether to read transcript point layers from the CosMx dataset.
    keep_gene_names
        Gene names to retain in transcript point layers and the upstream counts
        table. This can be a gene name, an iterable of gene names, or a path to a
        delimited file whose first column contains the gene panel, such as
        ``COAD_panel.csv``. If ``None``, no gene filtering is performed.
        Stored transcript point layers use Sparrow's canonical ``gene`` column
        name, rather than the upstream CosMx ``target`` column name.
    imread_kwargs
        Keyword arguments passed to the upstream image reader.
    image_models_kwargs
        Keyword arguments passed to the upstream image models.
    output
        Path where the resulting ``SpatialData`` object is backed. If ``None``,
        the result is returned in memory.

    Returns
    -------
    SpatialData
        The loaded CosMx data with elements named for their target coordinate
        system, for example ``1_points_global`` and ``table_global``.

    Raises
    ------
    ValueError
        If the number of paths and coordinate systems differs, coordinate
        systems are duplicated, dataset identifiers have the wrong length, or a
        transcript point layer lacks the CosMx target column.

    Examples
    --------
    >>> sdata = cosmx("/data/cosmx", keep_gene_names="COAD_panel.csv")
    """
    paths = _as_list(path)
    coordinate_systems = _as_list(to_coordinate_system)

    # Validate that every dataset has exactly one target coordinate system.
    if len(paths) != len(coordinate_systems):
        raise ValueError("The number of paths and coordinate systems must be equal.")

    # Reject duplicate coordinate systems because output layer names use them as suffixes.
    if len(coordinate_systems) != len(set(coordinate_systems)):
        raise ValueError("All coordinate systems must be unique.")

    # Reuse one dataset identifier for every path when a scalar was provided.
    if dataset_id is None:
        dataset_ids = [None] * len(paths)
    elif isinstance(dataset_id, str):
        dataset_ids = [dataset_id] * len(paths)
    else:
        dataset_ids = list(dataset_id)

    # Validate the per-path dataset identifiers before reading any data.
    if len(dataset_ids) != len(paths):
        raise ValueError("The number of dataset identifiers must match the number of paths.")

    log.info(
        "Starting CosMx read for %d dataset(s); transcripts=%s; coordinate systems=%s.",
        len(paths),
        transcripts,
        coordinate_systems,
    )

    # Load the optional keep list once so it is shared by all datasets.
    keep_genes = _load_keep_gene_names(keep_gene_names)

    if keep_genes is not None:
        log.info("Keeping %d CosMx genes while reading transcripts.", len(keep_genes))
    else:
        log.info("No CosMx gene whitelist supplied; no gene filtering will be applied.")

    sdata = SpatialData()

    # Initialize the requested backing store before loading potentially large transcript layers.
    if output is not None:
        log.info("Creating backed CosMx output at '%s'.", output)
        sdata.write(output)
        sdata = read_zarr(output)

    # Read and normalize each CosMx dataset independently.
    for dataset_index, (source_path, coordinate_system, source_dataset_id) in enumerate(
        zip(
            paths,
            coordinate_systems,
            dataset_ids,
            strict=True,
        ),
        start=1,
    ):
        log.info(
            "Reading CosMx dataset %d/%d from '%s' into coordinate system '%s'.",
            dataset_index,
            len(paths),
            source_path,
            coordinate_system,
        )

        # Delegate raw CosMx parsing and FOV-specific transforms to the spatialdata-io cosmx reader
        # The temporary source_sdata object will be discarded after each dataset as all the ellements
        # will be adjusted to comply with Sparrow's conventions and finally written to a common sdata object.
        source_sdata = sdata_cosmx(
            path=source_path,
            dataset_id=source_dataset_id,
            transcripts=transcripts,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
        )

        log.info(
            "Loaded raw CosMx dataset %d/%d: %d image(s), %d label(s), %d point layer(s), and %d table(s).",
            dataset_index,
            len(paths),
            len(source_sdata.images),
            len(source_sdata.labels),
            len(source_sdata.points),
            len(source_sdata.tables),
        )

        # Track destination names so a backed sdata object can persist each new element.
        destination_names: list[str] = []

        # Convert the image layers to have the appropriate transformation class for Sparrow.
        for layer_name in [*source_sdata.images]:
            # The combination of the source layer name and the target coordinate system is guaranteed to be unique
            #  because coordinate systems are validated to be unique.
            # This avoids overwriting the source layer when multiple datasets are read into the same SpatialData object.
            destination_name = f"{layer_name}_{coordinate_system}"
            _convert_spatial_element(
                sdata=sdata,
                source_sdata=source_sdata,
                layer_name=layer_name,
                destination_name=destination_name,
                coordinate_system=coordinate_system,
                keep_genes=None,
            )
            destination_names.append(destination_name)

        # Convert the label layers to have the appropriate transformation class for Sparrow.
        for layer_name in [*source_sdata.labels]:
            destination_name = f"{layer_name}_{coordinate_system}"
            _convert_spatial_element(
                sdata=sdata,
                source_sdata=source_sdata,
                layer_name=layer_name,
                destination_name=destination_name,
                coordinate_system=coordinate_system,
                keep_genes=None,
            )
            destination_names.append(destination_name)

        # Convert the point layers to have the appropriate transformation class for Sparrow
        # and optionally filter their gene variables to the requested keep list.
        for layer_name in [*source_sdata.points]:
            destination_name = f"{layer_name}_{coordinate_system}"
            _convert_spatial_element(
                sdata=sdata,
                source_sdata=source_sdata,
                layer_name=layer_name,
                destination_name=destination_name,
                coordinate_system=coordinate_system,
                keep_genes=keep_genes,
            )
            destination_names.append(destination_name)

        # Copy tables, filtering their gene variables and updating label-region references.
        for layer_name in [*source_sdata.tables]:
            destination_name = f"{layer_name}_{coordinate_system}"
            table = _prepare_table(
                source_sdata[layer_name],
                coordinate_system=coordinate_system,
                keep_genes=keep_genes,
            )
            sdata[destination_name] = table
            destination_names.append(destination_name)

        # Persist the current dataset before reading the next one when output is backed.
        if sdata.is_backed():
            sdata.write_element(destination_names)
            sdata = read_zarr(sdata.path)

    log.info(
        "Finished CosMx read: %d image(s), %d label(s), %d point layer(s), and %d table(s).",
        len(sdata.images),
        len(sdata.labels),
        len(sdata.points),
        len(sdata.tables),
    )

    return sdata


def _as_list(value: Any) -> list[Any]:
    """Return a scalar or iterable as a list without splitting strings or paths."""
    if isinstance(value, (str, Path)) or not isinstance(value, Iterable):
        return [value]

    return list(value)


def _load_keep_gene_names(
    keep_gene_names: str | Path | Iterable[str] | None,
) -> set[str] | None:
    """Load gene names from a sequence or the first column of a panel file."""
    if keep_gene_names is None:
        return None

    # Treat an explicit Path or an existing string path as a panel file.
    if isinstance(keep_gene_names, Path) or (
        isinstance(keep_gene_names, str) and Path(keep_gene_names).is_file()
    ):
        # Read the first column so one-column panels with arbitrary headers are supported.
        panel = pd.read_csv(keep_gene_names, header=0)

        # Reject empty files because they would silently remove every transcript.
        if panel.shape[1] == 0:
            raise ValueError("The CosMx gene panel file does not contain a column of gene names.")

        # Drop missing entries and normalize values to strings for Dask membership filtering.
        gene_names = set(panel.iloc[:, 0].dropna().astype(str))

        # Reject header-only panels because an empty whitelist would silently remove every gene.
        if not gene_names:
            raise ValueError("The CosMx gene panel file does not contain any gene names.")

        return gene_names

    # Normalize iterable gene names to strings for exact matching against CosMx targets.
    return {str(gene_name) for gene_name in keep_gene_names}


def _convert_spatial_element(
    sdata: SpatialData,
    source_sdata: SpatialData,
    layer_name: str,
    destination_name: str,
    coordinate_system: str,
    keep_genes: set[str] | None,
) -> None:
    """
    Normalize one upstream CosMx element and register it in the output object.

    The element is retrieved from ``source_sdata``, normalized for Sparrow's
    coordinate-system conventions, and stored in ``sdata`` under ``destination_name``.

    Image and label layers retain their existing pixel or
    label coordinates while **translation-only affine transformations** are
    represented as ``Translation`` objects. 
    
    Point layers are optionally filtered using ``keep_genes``,
    have the upstream ``target`` column renamed to Sparrow's canonical ``gene`` column, 
    and are transformed from their FOV-local coordinates into global coordinates.
    Because of this materialization into the global space, 
    their registered transformation can be set to ``Identity`` relative to ``coordinate_system``.
    This is because Sparrow requires identity-transformed points.
    """
    element = source_sdata[layer_name]

    # Retrieve the existing CosMx FOV-local-to-global transform before normalizing the element.
    global_transformation = get_transformation(element, to_coordinate_system="global")

    if layer_name in source_sdata.points:
        # Use the upstream CosMx target column while inspecting the source schema.
        target_column = CosmxKeys.TARGET_OF_TRANSCRIPT.value

        # Fail early if a custom upstream reader returns an unexpected point schema.
        if target_column not in element.columns:
            raise ValueError(f"CosMx point layer '{layer_name}' does not contain a '{target_column}' column.")

        # Apply the Dask dataframe membership filter without materializing the points.
        if keep_genes is not None:
            element = element[element[target_column].isin(keep_genes)]

        # Rename the upstream feature column so Sparrow's default allocation and plotting APIs work.
        element = element.rename(columns={target_column: _GENES_KEY})

        # Keep the PointsModel feature-key metadata synchronized with the renamed dataframe column.
        element.attrs["spatialdata_attrs"]["feature_key"] = _GENES_KEY

        # Numerically materialize the FOV transform in point coordinates because Sparrow requires identity-transformed points.
        element = _transform_points_to_global(element, global_transformation)

        # Mark the already-transformed points as identity-related to the requested coordinate system.
        global_transformation = Identity()
    else:
        # Convert translation-only Affines to the transformation type Sparrow's raster operations support.
        global_transformation = _normalize_transformation(global_transformation)

    # Register the normalized element under the requested coordinate-system name.
    # set_all=True replaces the previous transformation dictionary
    set_transformation(element, transformation={coordinate_system: global_transformation}, set_all=True)

    # Store the normalized element under a name that is unique to its coordinate system.
    sdata[destination_name] = element


def _normalize_transformation(transformation: Any) -> Any:
    """Convert **translation-only affine transforms** to Sparrow-compatible translations.

    Sparrow's allocation code currently understands ``Identity``, ``Translation``,
    and ``Sequence``, but not arbitrary ``Affine`` transformations.
    """
    # Leave non-affine transformations unchanged because they already carry their intended semantics.
    if not isinstance(transformation, Affine):
        return transformation

    # Express the affine transform in the two spatial axes used by CosMx images, labels, and points.
    affine_matrix = np.asarray(
        transformation.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
    )

    # Check whether the linear part is identity (i.e., leave x and y unchanged)
    # and the homogeneous row is present indicating a pure translation
    is_translation = np.allclose(affine_matrix[:2, :2], np.eye(2)) and np.allclose(
        affine_matrix[2], np.array([0.0, 0.0, 1.0])
    )

    # If the above check is false, it indicates genuine scale, rotation, or shear transforms
    # That cannot be represented by a simple translation, so we return the original Affine.
    if not is_translation:
        return transformation

    # Represent the same translation using the class accepted by Sparrow's raster-coordinate helpers.
    # affine_matrix[:2, -1] extracts the translation vector from the affine matrix
    return Translation(affine_matrix[:2, -1], axes=("x", "y"))


def _transform_points_to_global(points: dd.DataFrame, transformation: Any) -> dd.DataFrame:
    """Apply a CosMx FOV transform to point coordinates and return identity-transformed points."""
    # Convert the SpatialData transformation into a two-dimensional homogeneous affine matrix.
    affine_matrix = np.asarray(
        transformation.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
    )

    # Define the partition operation so Dask can apply the transform lazily to transcript chunks.
    def apply_affine(partition: pd.DataFrame) -> pd.DataFrame:
        # Read local FOV coordinates
        local_coordinates = partition[["x", "y"]].to_numpy()
        # Append the homogeneous coordinate to enable translation through matrix multiplication with the affine matrix
        homogeneous_coordinates = np.column_stack((local_coordinates, np.ones(len(partition))))

        # Apply the FOV-to-global transform to every point in the partition.
        global_coordinates = homogeneous_coordinates @ affine_matrix.T

        # Copy the partition so the source point dataframe and its lazy graph remain unchanged.
        transformed_partition = partition.copy()

        # Replace local x and y values with their calculated global-coordinate equivalents.
        transformed_partition["x"] = global_coordinates[:, 0]
        transformed_partition["y"] = global_coordinates[:, 1]

        return transformed_partition

    # Update the empty metadata frame so Dask records the transformed coordinate dtypes correctly.
    metadata = points._meta.copy()
    metadata["x"] = metadata["x"].astype(float)
    metadata["y"] = metadata["y"].astype(float)

    # Build a lazy transformed dataframe without computing the full transcript table.
    return points.map_partitions(apply_affine, meta=metadata)


def _prepare_table(
    table: Any,
    coordinate_system: str,
    keep_genes: set[str] | None,
) -> Any:
    """Filter and rename the region metadata in an upstream CosMx table."""
    # Select only requested genes while preserving all cell observations and metadata.
    if keep_genes is not None:
        table = table[:, table.var_names.isin(keep_genes)].copy()

    # Validate the columns needed to keep table regions linked to renamed label layers.
    if _REGION_KEY not in table.obs.columns:
        raise ValueError(f"CosMx table does not contain the region column '{_REGION_KEY}'.")
    if _INSTANCE_KEY not in table.obs.columns:
        raise ValueError(f"CosMx table does not contain the instance column '{_INSTANCE_KEY}'.")

    # Append the target coordinate system to every referenced label region.
    # This is needed because the label layers were renamed to include the coordinate system.
    region_names = table.obs[_REGION_KEY].astype(str).map(lambda region: f"{region}_{coordinate_system}")
    # Store the region values as categorical data, which is the representation expected by TableModel.
    table.obs[_REGION_KEY] = pd.Categorical(region_names)

    # Remove the SpatialData table metadata so it can be rebuilt with the new region names.
    table.uns.pop(TableModel.ATTRS_KEY, None)

    # Rebuild SpatialData table metadata after changing categorical region values.
    return TableModel.parse(
        table,
        region_key=_REGION_KEY,
        region=table.obs[_REGION_KEY].cat.categories.to_list(),
        instance_key=_INSTANCE_KEY,
    )
