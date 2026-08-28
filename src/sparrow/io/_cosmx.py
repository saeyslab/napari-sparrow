from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from dask.array import Array
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Multiscales, Reader
from scipy.sparse import csr_matrix
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity
from spatialdata.transformations._utils import _set_transformations, compute_coordinates
from xarray import DataTree

from sparrow.image._manager import ImageLayerManager, LabelLayerManager
from sparrow.points._points import add_points_layer
from sparrow.table._table import add_table_layer
from sparrow.utils._keys import _GENES_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# Common CosMx export suffixes. Prefer parquet transcripts when both formats exist.
_TRANSCRIPT_SUFFIXES = ("_tx_file.parquet", "_tx_file.csv")
_FOV_POSITIONS_SUFFIXES = ("_fov_positions_file.csv",)
_COUNTS_SUFFIXES = ("_exprMat_file.csv",)
_METADATA_SUFFIXES = ("_metadata_file.csv",)

# Named image folders used by AtoMx / CosMx exports. CellLabels is never treated as an image folder.
_IMAGE_DIR_CANDIDATES = ("CellComposite", "Morphology2D", "CellOverlay", "Morphology")
_LABEL_DIR_NAME = "CellLabels"

# Transcript and FOV-position column aliases seen across CosMx export versions.
_GENE_COLUMNS = ("target", "gene")
_GLOBAL_X_COLUMNS = ("x_global_px", "X_global_px")
_GLOBAL_Y_COLUMNS = ("y_global_px", "Y_global_px")
_LOCAL_X_COLUMNS = ("x_local_px", "X_local_px")
_LOCAL_Y_COLUMNS = ("y_local_px", "Y_local_px")
_FOV_COLUMNS = ("fov", "FOV")
_FOV_X_COLUMNS = ("x_global_px", "X_global_px", "x_px", "X_px")
_FOV_Y_COLUMNS = ("y_global_px", "Y_global_px", "y_px", "Y_px")
_CELL_ID_COLUMNS = ("cell_ID", "cell_id")
_CENTER_X_GLOBAL_COLUMNS = ("CenterX_global_px",)
_CENTER_Y_GLOBAL_COLUMNS = ("CenterY_global_px",)

_BLOCKSIZE = "256MB"


@dataclass(frozen=True)
class _CosmxFiles:
    """Resolved files for one CosMx dataset root."""

    transcripts: Path
    images_dir: Path
    fov_positions: Path | None
    labels_dir: Path | None
    counts: Path | None
    metadata: Path | None


@dataclass(frozen=True)
class _CosmxDataset:
    """Validated global Zarr stores and transcript metadata for one CosMx dataset root."""

    path: Path
    transcripts_file: Path
    counts_file: Path | None
    metadata_file: Path | None
    image_store: Path
    labels_store: Path | None
    transcripts: dd.DataFrame


def cosmx(
    path: str | Path | list[str] | list[Path],
    to_coordinate_system: str | list[str] = "global",
    dataset_id: str | list[str] | None = None,
    keep_gene_names: str | Path | Iterable[str] | None = None,
    cells_labels: bool = False,
    cells_table: bool = False,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    output: str | Path | None = None,
) -> SpatialData:
    """Read *CosMx* data into a ``SpatialData`` object.

    Images and transcripts are required. Vendor cell labels and the vendor
    cell-by-gene table are optional. Transcripts are read lazily into a single
    points layer using global pixel coordinates when those columns exist.
    Gene names can be filtered to a whitelist before any coordinate work is performed.
    Vendor image and label OME-Zarr pyramids are already global dataset-level
    data and are therefore each loaded once with an identity transformation.
    A ``fov_positions_file.csv`` is used only when local transcript coordinates
    need to be translated into the global pixel system.

    Parameters
    ----------
    path
        Path to a CosMx dataset root, or a list of roots to combine.
    to_coordinate_system
        Coordinate system assigned to each dataset. If a list is provided, its
        length must match ``path`` and every name must be unique.
    dataset_id
        Dataset identifier used to select ``<dataset_id>_tx_file.csv`` and related
        files. A scalar identifier is reused for every path; a list must match
        the length of ``path``. If ``None``, the identifier is inferred from the
        transcript file name.
    keep_gene_names
        Gene names to retain in the transcript layer and, when requested, the
        vendor table. This can be an iterable of gene names or a path to a
        delimited file whose first column contains the gene panel.
        String values are interpreted as panel file paths.
        If ``None``, no gene filtering is performed.
        Stored transcript layers use Sparrow's canonical ``gene`` column.
    cells_labels
        Whether to read vendor ``CellLabels`` masks. Automatically enabled when
        ``cells_table`` is ``True`` because the cell table is linked to the vendor label masks.
    cells_table
        Whether to read the vendor ``exprMat`` counts and cell metadata table.
    image_models_kwargs
        Keyword arguments controlling native image and label pyramid chunks.
        The only supported key is ``chunks``; vendor pyramid levels are always
        retained without generating replacement levels.
    output
        Path where the resulting ``SpatialData`` object is backed. If ``None``,
        the result is returned in memory.

    Returns
    -------
    SpatialData
        CosMx images as ``image_{coordinate_system}`` and transcripts as
        ``transcripts_{coordinate_system}``. Optional labels and tables use
        ``labels_{coordinate_system}`` and ``table_{coordinate_system}``.

    Raises
    ------
    ValueError
        If paths and coordinate systems are mismatched, coordinate systems are
        duplicated, or ``keep_gene_names`` is a scalar gene name or empty whitelist.
    FileNotFoundError
        If a required image directory, transcript file, requested vendor
        labels/table file, or ``Path`` gene panel cannot be found.

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

    # Validate the per-path dataset identifiers before reading any data.
    if dataset_id is None:
        dataset_ids = [None] * len(paths)
    elif isinstance(dataset_id, str):
        dataset_ids = [dataset_id] * len(paths)
    else:
        dataset_ids = list(dataset_id)
        if len(dataset_ids) != len(paths):
            raise ValueError("The number of dataset identifiers must match the number of paths.")

    # Warn about model options that this reader cannot apply to native vendor pyramids.
    unsupported_image_model_keys = [str(key) for key in image_models_kwargs if key != "chunks"]
    if unsupported_image_model_keys:
        log.warning(
            "Ignoring unsupported 'image_models_kwargs' keys: %s. Supported key is 'chunks'.",
            ", ".join(sorted(unsupported_image_model_keys))
        )

    # Vendor tables annotate label layers, so reading the table implies reading labels.
    if cells_table and not cells_labels:
        log.info("Setting 'cells_labels' to True so the vendor table can annotate CosMx label layers.")
        cells_labels = True

    # Load the optional keep list once so it is shared by all datasets.
    keep_genes = _load_keep_gene_names(keep_gene_names)
    if keep_genes is not None:
        log.info("Keeping %d CosMx genes while reading transcripts.", len(keep_genes))
    else:
        log.info("No CosMx gene whitelist supplied; no gene filtering will be applied.")

    log.info(
        "Starting CosMx reader for %d dataset(s); cells_labels=%s; cells_table=%s; coordinate systems=%s.",
        len(dataset_ids),
        cells_labels,
        cells_table,
        coordinate_systems,
    )

    # Validate every dataset before creating an output store so invalid later datasets cannot leave partial output.
    validated_datasets: list[_CosmxDataset] = []
    for source_path, coordinate_system, source_dataset_id in zip(
        paths,
        coordinate_systems,
        dataset_ids,
        strict=True,
    ):
        log.info(
            "Validating CosMx dataset from '%s' for coordinate system '%s'.",
            source_path,
            coordinate_system,
        )
        validated_datasets.append(
            _prepare_dataset(
                path=Path(source_path),
                dataset_id=source_dataset_id,
                keep_genes=keep_genes,
                cells_labels=cells_labels,
                cells_table=cells_table,
            )
        )

    sdata = SpatialData()

    # Initialize the backing store before reading transcripts so points can spill to disk.
    if output is not None:
        log.info("Creating zarr store for spatial data backing at '%s'.", output)
        sdata.write(output)
        sdata = read_zarr(output)

    # Add each validated dataset to the output store with its requested coordinate system.
    for dataset_index, (dataset, coordinate_system) in enumerate(
        zip(validated_datasets, coordinate_systems, strict=True),
        start=1,
    ):
        log.info(
            "Reading CosMx dataset %d/%d from '%s' into coordinate system '%s'.",
            dataset_index,
            len(validated_datasets),
            dataset.path,
            coordinate_system,
        )
        sdata = _add_dataset(
            sdata,
            dataset=dataset,
            coordinate_system=coordinate_system,
            keep_genes=keep_genes,
            cells_labels=cells_labels,
            cells_table=cells_table,
            image_models_kwargs=image_models_kwargs,
        )

    log.info(
        "Finished CosMx read: %d image(s), %d label(s), %d point layer(s), and %d table(s).",
        len(sdata.images),
        len(sdata.labels),
        len(sdata.points),
        len(sdata.tables),
    )

    return sdata


def _prepare_dataset(
    path: Path,
    dataset_id: str | None,
    keep_genes: set[str] | None,
    cells_labels: bool,
    cells_table: bool,
) -> _CosmxDataset:
    """Discover and validate one CosMx dataset root."""
    # Find and validate all relevant dataset files and Zarr stores for this dataset
    dataset_files = _discover_files(path, dataset_id=dataset_id, cells_labels=cells_labels, cells_table=cells_table)

    # Require that the discovered images directory is an OME-Zarr group rather than a legacy raster directory.
    if not _is_zarr_group(dataset_files.images_dir):
        raise FileNotFoundError(
            f"CosMx image store '{dataset_files.images_dir}' is not an OME-Zarr group. Legacy raster inputs are not supported."
        )

    # Require that the discovered labels directory is an OME-Zarr group when labels are requested.
    if cells_labels and dataset_files.labels_dir is not None and not _is_zarr_group(dataset_files.labels_dir):
        raise FileNotFoundError(
            f"CosMx label store '{dataset_files.labels_dir}' is not an OME-Zarr group. Legacy raster inputs are not supported."
        )

    # Load FOV origins from the optional FOV positions file
    # so local transcript coordinates can be translated to global pixel coordinates.
    fov_origins = _load_fov_origins(dataset_files.fov_positions)

    # Read transcripts lazily from parquet or CSV.
    if dataset_files.transcripts.suffix.lower() == ".parquet":
        transcripts = dd.read_parquet(dataset_files.transcripts)
    else:
        transcripts = dd.read_csv(dataset_files.transcripts, header=0, blocksize=_BLOCKSIZE)

    # Require a recognized gene column name in the transcripts table.
    gene_column = _require_column(transcripts.columns, _GENE_COLUMNS, kind="gene")
    if keep_genes is not None:
        # Filter transcripts lazily to the specified whitelist before coordinate transformations.
        transcripts = transcripts[transcripts[gene_column].isin(keep_genes)]

    # Normalize transcript coordinates, handling global vs local FOV placement lazily.
    transcripts = _prepare_transcripts(
        transcripts=transcripts,
        transcripts_path=dataset_files.transcripts,
        gene_column=gene_column,
        fov_origins=fov_origins,
        fov_positions_path=dataset_files.fov_positions,
    )

    return _CosmxDataset(
        path=path,
        transcripts_file=dataset_files.transcripts,
        image_store=dataset_files.images_dir,
        labels_store=dataset_files.labels_dir,
        counts_file=dataset_files.counts,
        metadata_file=dataset_files.metadata,
        transcripts=transcripts,
    )


def _add_dataset(
    sdata: SpatialData,
    dataset: _CosmxDataset,
    coordinate_system: str,
    keep_genes: set[str] | None,
    cells_labels: bool,
    cells_table: bool,
    image_models_kwargs: Mapping[str, Any],
) -> SpatialData:
    """Add one validated CosMx dataset to ``sdata``."""
    chunks = image_models_kwargs.get("chunks")

    # Process the multiscale image and register it with SpatialData.
    image_tree = _process_multiscale_raster(
        dataset.image_store,
        coordinate_system=coordinate_system,
        kind="image",
        chunks=chunks,
    )
    sdata = ImageLayerManager().add_to_sdata(
        sdata,
        output_layer=f"image_{coordinate_system}",
        spatial_element=image_tree,
        overwrite=False,
    )

    # Add lazily evaluated transcript points with an identity coordinate transform.
    sdata = add_points_layer(
        sdata,
        ddf=dataset.transcripts,
        output_layer=f"transcripts_{coordinate_system}",
        coordinates={"x": "x", "y": "y"},
        transformations={coordinate_system: Identity()},
        overwrite=False,
    )

    if cells_labels:
        assert dataset.labels_store is not None
        # Process multiscale integer label masks and register with SpatialData.
        labels_tree = _process_multiscale_raster(
            dataset.labels_store,
            coordinate_system=coordinate_system,
            kind="labels",
            chunks=chunks,
        )
        sdata = LabelLayerManager().add_to_sdata(
            sdata,
            output_layer=f"labels_{coordinate_system}",
            spatial_element=labels_tree,
            overwrite=False,
        )

    if cells_table:
        sdata = _add_table(
            sdata,
            counts_path=dataset.counts_file,
            metadata_path=dataset.metadata_file,
            coordinate_system=coordinate_system,
            keep_genes=keep_genes,
        )

    return sdata


def _discover_files(
    path: Path,
    dataset_id: str | None,
    cells_labels: bool,
    cells_table: bool,
) -> _CosmxFiles:
    """Locate and validate the standard CosMx files under ``path``."""
    # Check for the required transcript file through the known suffixes, preferring parquet over CSV when both exist.
    transcripts = _find_suffix_file(path, dataset_id, _TRANSCRIPT_SUFFIXES)
    if transcripts is None:
        raise FileNotFoundError(
            f"CosMx transcript file not found in '{path}'. Expected a file ending with {', '.join(_TRANSCRIPT_SUFFIXES)}."
        )

    # Check for the required image directory through the known candidates or any nested Zarr group.
    images_dir = _find_directory(path, _IMAGE_DIR_CANDIDATES, skip_names={_LABEL_DIR_NAME})
    if images_dir is None:
        raise FileNotFoundError(
            f"CosMx image Zarr directory not found in '{path}'. Expected one of {', '.join(_IMAGE_DIR_CANDIDATES)}."
        )

    # Resolve and validate the labels directory only when labels were requested.
    labels_dir: Path | None = None
    if cells_labels:
        labels_dir = path / _LABEL_DIR_NAME
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"CosMx labels directory not found in '{path}'. Expected '{_LABEL_DIR_NAME}'.")

    # Infer one dataset identifier from the required transcript file so related files cannot come from another dataset.
    if dataset_id is None:
        current_transcript_suffix = next(suffix for suffix in _TRANSCRIPT_SUFFIXES if transcripts.name.endswith(suffix))
        # Derive the shared dataset prefix from the transcript filename selected above.
        resolved_dataset_id = transcripts.name[: -len(current_transcript_suffix)]
    else:
        resolved_dataset_id = dataset_id

    # Find the optional FOV positions file, counts file, and metadata file through their known suffixes.
    fov_positions = _find_suffix_file(path, resolved_dataset_id, _FOV_POSITIONS_SUFFIXES)
    counts = _find_suffix_file(path, resolved_dataset_id, _COUNTS_SUFFIXES)
    metadata = _find_suffix_file(path, resolved_dataset_id, _METADATA_SUFFIXES)
    # Require both counts and metadata when the vendor table is requested, otherwise ignore missing files.
    if cells_table and (counts is None or metadata is None):
        raise FileNotFoundError(
            f"CosMx counts/metadata files not found in '{path}' which is required when 'cells_table' is True. "
            f"Expected files ending with {', '.join(_COUNTS_SUFFIXES)} and {', '.join(_METADATA_SUFFIXES)}."
        )

    return _CosmxFiles(
        transcripts=transcripts,
        images_dir=images_dir,
        fov_positions=fov_positions,
        labels_dir=labels_dir,
        counts=counts,
        metadata=metadata,
    )


def _prepare_transcripts(
    transcripts: dd.DataFrame,
    transcripts_path: Path,
    gene_column: str,
    fov_origins: Mapping[str, tuple[float, float]] | None,
    fov_positions_path: Path | None,
) -> dd.DataFrame:
    """Normalize, place, and validate CosMx transcript coordinates."""
    # Check if global pixel coordinates are present in the transcript file.
    global_x = _optional_column(transcripts.columns, _GLOBAL_X_COLUMNS)
    global_y = _optional_column(transcripts.columns, _GLOBAL_Y_COLUMNS)

    if global_x is not None and global_y is not None:
        log.info("Using CosMx global pixel columns '%s' and '%s' for transcript coordinates.", global_x, global_y)
        # Rename global coordinate columns to canonical names for Sparrow points layers.
        transcripts = transcripts.rename(columns={gene_column: _GENES_KEY, global_x: "x", global_y: "y"})
    else:
        # Validate required local coordinate columns and FOV column when global coordinates are absent.
        local_x = _require_column(transcripts.columns, _LOCAL_X_COLUMNS, kind="local x")
        local_y = _require_column(transcripts.columns, _LOCAL_Y_COLUMNS, kind="local y")
        fov_column = _require_column(transcripts.columns, _FOV_COLUMNS, kind="fov")

        # Apply FOV origins to local coordinates when FOV placement metadata is available.
        if fov_origins is not None:
            log.info("Global transcript coordinates are absent; applying FOV translations from fov_positions.")
            # Create meta schema with columns and dtypes returned by the partition-level coordinate transformation.
            meta = transcripts._meta.rename(columns={gene_column: _GENES_KEY, local_x: "x", local_y: "y"})
            meta[_GENES_KEY] = meta[_GENES_KEY].astype(str)
            meta["x"] = meta["x"].astype(float)
            meta["y"] = meta["y"].astype(float)

            # Use Dask's map_partitions to apply FOV origins to local coordinates in a fully vectorized manner.
            # Dask builds a task graph for each partition for which it needs to know some metadata
            # about the output DataFrame, so we provide a meta DataFrame with the expected columns and dtypes.
            transcripts = transcripts.map_partitions(
                _apply_fov_origins,
                gene_column=gene_column,
                local_x=local_x,
                local_y=local_y,
                fov_column=fov_column,
                fov_origins=fov_origins,
                meta=meta,
            )
        else:
            if fov_positions_path is None:
                positions_description = "no FOV positions file was found"
            else:
                positions_description = f"FOV positions file '{fov_positions_path}' has no usable pixel origins"
            raise ValueError(
                f"CosMx transcript file '{transcripts_path}' contains local FOV coordinates, but "
                f"{positions_description}; global transcript coordinates cannot be determined."
            )

    transcripts["x"] = transcripts["x"].astype(float)
    transcripts["y"] = transcripts["y"].astype(float)

    return transcripts


def _add_table(
    sdata: SpatialData,
    counts_path: Path,
    metadata_path: Path,
    coordinate_system: str,
    keep_genes: set[str] | None,
) -> SpatialData:
    """Read the optional vendor cell-by-gene table."""
    log.info("Reading CosMx vendor table from '%s' and '%s'.", counts_path, metadata_path)

    counts = pd.read_csv(counts_path, header=0)
    metadata = pd.read_csv(metadata_path, header=0)

    counts_fov = _require_column(counts.columns, _FOV_COLUMNS, kind="fov")
    counts_cell = _require_column(counts.columns, _CELL_ID_COLUMNS, kind="cell ID")
    metadata_fov = _require_column(metadata.columns, _FOV_COLUMNS, kind="fov")
    metadata_cell = _require_column(metadata.columns, _CELL_ID_COLUMNS, kind="cell ID")

    # Build a unique observation index that is stable across FOVs.
    counts_index = _cell_index(counts[counts_cell], counts[counts_fov])
    metadata_index = _cell_index(metadata[metadata_cell], metadata[metadata_fov])
    counts = counts.set_index(counts_index)
    metadata = metadata.set_index(metadata_index)

    # Drop identifier columns from the expression matrix so remaining columns are genes.
    gene_columns = [column for column in counts.columns if column not in {counts_fov, counts_cell}]
    counts = counts[gene_columns]

    common_index = metadata.index.intersection(counts.index)
    counts = counts.loc[common_index]
    metadata = metadata.loc[common_index].copy()

    if keep_genes is not None:
        counts = counts.loc[:, counts.columns.astype(str).isin(keep_genes)]

    # Point every vendor cell at the single global labels layer stored in this coordinate system.
    metadata[_REGION_KEY] = pd.Categorical([f"labels_{coordinate_system}"] * len(metadata), ordered=False)
    metadata[_INSTANCE_KEY] = metadata[metadata_cell].astype(np.int64)

    adata = AnnData(
        X=csr_matrix(counts.to_numpy()),
        obs=metadata,
        var=pd.DataFrame(index=counts.columns.astype(str)),
    )

    # Preserve global cell centres when the vendor metadata provides them.
    center_x = _optional_column(adata.obs.columns, _CENTER_X_GLOBAL_COLUMNS)
    center_y = _optional_column(adata.obs.columns, _CENTER_Y_GLOBAL_COLUMNS)
    if center_x is not None and center_y is not None:
        adata.obsm[_SPATIAL] = adata.obs[[center_x, center_y]].to_numpy()

    return add_table_layer(
        sdata,
        adata=adata,
        output_layer=f"table_{coordinate_system}",
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=False,
    )


def _as_list(value: Any) -> list[Any]:
    """Return a scalar or iterable as a list without splitting strings or paths."""
    if isinstance(value, (str, Path)) or not isinstance(value, Iterable):
        return [value]

    return list(value)


def _load_keep_gene_names(keep_gene_names: str | Path | Iterable[str] | None) -> set[str] | None:
    """Load gene names from an iterable or the first column of a panel file."""
    if keep_gene_names is None:
        return None

    # Treat an explicit Path or an existing string path as a panel file.
    panel_path: Path | None = None
    if isinstance(keep_gene_names, (str, Path)):
        panel_path = Path(keep_gene_names)
        if not panel_path.is_file():
            if isinstance(keep_gene_names, Path):
                raise FileNotFoundError(f"CosMx gene panel file does not exist: {panel_path}")
            raise ValueError(
                "`keep_gene_names` must be a path to a gene panel file or an iterable "
                "of gene names; single gene names are not supported."
            )

    if panel_path is not None:
        # Read the first column so one-column panels with arbitrary headers are supported.
        panel = pd.read_csv(panel_path, header=0)

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
    gene_names = {str(gene_name) for gene_name in keep_gene_names}
    if not gene_names:
        raise ValueError("The CosMx gene whitelist must contain at least one gene name.")

    return gene_names


def _find_suffix_file(path: Path, dataset_id: str | None, suffixes: tuple[str, ...]) -> Path | None:
    """Return an exact dataset file, or a unique suffix match when no ID was supplied."""
    # Use only exact filenames when the caller selected a dataset explicitly.
    if dataset_id is not None:
        for suffix in suffixes:
            candidate = path / f"{dataset_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    # Infer a file when no dataset_id was supplied.
    for suffix in suffixes:
        matches = sorted(child for child in path.iterdir() if child.is_file() and child.name.endswith(suffix))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple files ending with '{suffix}' found in '{path}'. Please specify dataset_id.")

    return None


def _find_directory(path: Path, candidates: tuple[str, ...], skip_names: set[str]) -> Path | None:
    """Return a named image folder, or a subdirectory containing a Zarr group."""
    for name in candidates:
        candidate = path / name
        if candidate.is_dir():
            return candidate

    # If no candidate folder matched, inspect subdirectories for an unnamed Zarr group.
    for child in sorted(path.iterdir()):
        # Accept unnamed global-Zarr directories without interpreting filenames as FOV identifiers.
        if child.is_dir() and child.name not in skip_names and _is_zarr_group(child):
            # Warn that an unnamed Zarr directory was detected and used instead of standard candidate folders.
            log.warning(
                "No standard image directory found in '%s' (%s); continuing with discovered Zarr store '%s'.",
                path,
                ", ".join(candidates),
                child.name,
            )
            return child

    return None


def _is_zarr_group(path: Path) -> bool:
    """Return whether a path contains local Zarr group metadata."""
    # Support both Zarr v2 and v3 group metadata markers without opening arbitrary directories.
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def _process_multiscale_raster(
    path: Path,
    coordinate_system: str,
    kind: str,
    chunks: str | tuple[int, ...] | int | None = None,
) -> DataTree:
    """Process a channel-grouped image or multiscale labels CosMx OME-Zarr store into a SpatialData-compatible DataTree."""
    if kind == "image":
        # Open container metadata to discover image channel sub-groups.
        image_group = zarr.open_group(path, mode="r")
        channel_names = sorted(name for name, _ in image_group.groups())
        if not channel_names:
            raise ValueError(f"CosMx image Zarr store '{path}' does not contain channel groups.")

        # Read native pyramid levels lazily for each channel store.
        channel_levels = [_read_cosmx_zarr_levels(path / channel_name, kind="image") for channel_name in channel_names]
        expected_level_count = len(channel_levels[0])
        # Validate that all channels have the same number of pyramid levels.
        if any(len(levels) != expected_level_count for levels in channel_levels[1:]):
            differing_counts = [len(levels) for levels in channel_levels]
            raise ValueError(
                f"CosMx image channels in '{path}' expose different numbers of pyramid levels: {differing_counts}."
            )

        # Stack corresponding native levels across channels along the c dimension.
        raster_levels: list[Array] = []
        for level_index in range(expected_level_count):
            # Get the arrays for the current level from each channel.
            level_channel_arrays = [channel[level_index] for channel in channel_levels]
            # Stack the channel arrays of the current level into a single lazy 3D array (c, y, x).
            multichannel_raster_level = da.stack(level_channel_arrays, axis=0)
            raster_levels.append(multichannel_raster_level)

        model = Image2DModel
        dims = ("c", "y", "x")
        c_coords = channel_names
    else:
        # Read multiscale labels lazily from the single labels store.
        # Cast native arrays to uint32 for compatibility with Sparrow’s segmentation utilities.
        raster_levels = [resolution.astype(np.uint32) for resolution in _read_cosmx_zarr_levels(path, kind="labels")]
        model = Labels2DModel
        dims = ("y", "x")
        c_coords = None

    return _build_multiscale_tree(
        arrays=raster_levels,
        coordinate_system=coordinate_system,
        model=model,
        dims=dims,
        channel_names=c_coords,
        chunks=chunks,
    )


def _read_cosmx_zarr_levels(path: Path, kind: str) -> list[Array]:
    """
    Return native 2D OME-Zarr levels as lazy Dask arrays.

    The function validates that a given CosMx Zarr group contains exactly one 2D OME-Zarr pyramid, 
    obtains the ordered level paths from OME-Zarr metadata, 
    and exposes those levels as lazy Dask arrays using direct Zarr access.
    ome-zarr.Reader is used to understand the OME-Zarr metadata and dataset ordering.
    zarr.open_group is used to access the actual arrays.
    """
    # We start by wrapping the local filesystem path in an OME-Zarr location object with ZarrLocation.
    # Then we parse the given Zarr instance into a collection of Nodes with the Reader
    nodes = list(Reader(ZarrLocation(path))())
    # Select the single multiscale node from the collection of nodes.
    multiscale_nodes = [node for node in nodes if any(isinstance(spec, Multiscales) for spec in node.specs)]
    # Distinguish missing multiscale metadata from an ambiguous store with multiple datasets.
    multiscale_node_count = len(multiscale_nodes)
    if multiscale_node_count == 0:
        raise ValueError(
            f"CosMx {kind} Zarr store '{path}' does not contain a readable multiscale OME-Zarr dataset."
        )
    if multiscale_node_count > 1:
        raise ValueError(
            f"CosMx {kind} Zarr store '{path}' contains {multiscale_node_count} multiscale nodes; "
            "expected exactly one."
        )

    # Use OME-Zarr multiscale metadata (i.e. the pyramid descriptor) for native dataset ordering
    node = multiscale_nodes[0]
    multiscales = node.load(Multiscales)
    # Check that the multiscale metadata was successfully loaded
    if multiscales is None:
        raise ValueError(f"CosMx {kind} Zarr store '{path}' has no readable multiscale metadata.")

    # Open the same store with Zarr so Dask reads the vendor chunks correctly on Windows.
    zarr_group = zarr.open_group(path, mode="r")
    # The order in multiscales.datasets matters because the OME-Zarr metadata defines which
    #  dataset is level zero, level one, etc.
    # Read each dataset lazily into a Dask array without changing the native pyramid levels.
    raster_levels = [da.from_zarr(zarr_group[dataset_path]) for dataset_path in multiscales.datasets]

    # Enforce that every level is a 2D array because CosMx images and labels should be 2D rasters.
    for level in raster_levels:
        if len(level.shape) != 2:
            raise ValueError(
                f"Level {level} of CosMx {kind} Zarr store '{path}' has shape {level.shape}; expected 2D arrays."
            )
    return raster_levels


def _build_multiscale_tree(
    arrays: list[Array],
    coordinate_system: str,
    model: Any,
    dims: tuple[str, ...],
    channel_names: list[str] | None,
    chunks: str | tuple[int, ...] | int | None,
) -> DataTree:
    """Parse lazy Dask arrays and build a DataTree structure expected by SpatialData for multiscale images or labels."""
    levels: dict[str, Any] = {}
    for level_index, array in enumerate(arrays):
        if chunks is not None:
            # Rechunk the array to the requested chunking scheme for better Dask performance.
            array = array.rechunk(chunks)

        # Convert the Dask array for each existing level into a SpatialData-compatible xarray.DataArray
        #  with the specified dimensions and channel coordinates.
        parsed = model.parse(
            array,
            dims=dims,
            c_coords=channel_names,
        )
        # Wrap the DataArray in an xarray.Dataset, compatible with SpatialData's expectations for DataTree nodes.
        levels[f"scale{level_index}"] = parsed.to_dataset(name="image")

    # Build the multiscale DataTree from the Datasets
    tree = DataTree.from_dict(levels)
    # The level-zero CosMx mosaic is already in the requested global pixel coordinate system,
    # so it receives an Identity() transform. 
    # For lower levels, SpatialData derives scale factors from the shapes.
    _set_transformations(tree, {coordinate_system: Identity()})

    # Compute every level's pixel centres in the level-zero CosMx pixel coordinate system, needed xarray coordinate computations
    return compute_coordinates(tree)


def _load_fov_origins(fov_positions_path: Path | None) -> dict[str, tuple[float, float]] | None:
    """Read FOV translations from ``fov_positions_file.csv`` when pixel columns exist."""
    if fov_positions_path is None:
        return None

    # Read the FOV positions file and identify the required columns for FOV, x, and y origins.
    positions = pd.read_csv(fov_positions_path, header=0)
    fov_column = _optional_column(positions.columns, _FOV_COLUMNS)
    x_column = _optional_column(positions.columns, _FOV_X_COLUMNS)
    y_column = _optional_column(positions.columns, _FOV_Y_COLUMNS)

    if fov_column is None or x_column is None or y_column is None:
        log.warning(
            "CosMx FOV positions file '%s' does not contain pixel origin columns; "
            "global image and label rasters will use identity transformations.",
            fov_positions_path,
        )
        return None

    origins: dict[str, tuple[float, float]] = {}
    for _, row in positions.iterrows():
        # Normalize the FOV key before checking for duplicate vendor rows.
        fov = _normalize_fov(row[fov_column])
        if fov in origins:
            raise ValueError(f"CosMx FOV positions file '{fov_positions_path}' contains duplicate FOV '{fov}'.")

        try:
            origin = (float(row[x_column]), float(row[y_column]))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"CosMx FOV positions file '{fov_positions_path}' contains a non-numeric origin for FOV '{fov}'."
            ) from error

        # Reject invalid origins before they become Translation transforms and corrupt allocation coordinates.
        if not all(np.isfinite(value) for value in origin):
            raise ValueError(
                f"CosMx FOV positions file '{fov_positions_path}' contains a non-finite origin for FOV '{fov}'."
            )

        origins[fov] = origin

    return origins


def _apply_fov_origins(
    partition: pd.DataFrame,
    gene_column: str,
    local_x: str,
    local_y: str,
    fov_column: str,
    fov_origins: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Add FOV origin translations to local transcript coordinates in a fully vectorized manner."""
    # Create origin coordinate lookup maps to avoid row-by-row lambda evaluation in Pandas.
    x_map = {fov: origin[0] for fov, origin in fov_origins.items()}
    y_map = {fov: origin[1] for fov, origin in fov_origins.items()}

    # Normalize FOVs vectorially and map to numeric origin offsets.
    fov_ids = partition[fov_column].astype("int64").astype(str)
    origin_x = fov_ids.map(x_map)
    origin_y = fov_ids.map(y_map)

    # Validate that every FOV in this partition has a corresponding FOV origin.
    if origin_x.isna().any() or origin_y.isna().any():
        missing_origin_fovs = sorted(fov_ids[origin_x.isna() | origin_y.isna()])
        raise ValueError(
            "CosMx FOV positions metadata is incomplete; missing origins for FOVs "
            f"{', '.join(missing_origin_fovs)}. Valid origins are required for every local transcript FOV."
        )

    # Preserve untouched vendor columns while replacing source columns with Sparrow's canonical names.
    translated = partition.rename(columns={gene_column: _GENES_KEY, local_x: "x", local_y: "y"}).copy()
    translated[_GENES_KEY] = partition[gene_column].astype(str)
    translated["x"] = partition[local_x].astype(float) + origin_x.astype(float)
    translated["y"] = partition[local_y].astype(float) + origin_y.astype(float)

    return translated


def _optional_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name, or ``None`` if none are present."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def _require_column(columns: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    """Return the first matching column name or raise a descriptive error."""
    column = _optional_column(columns, candidates)
    if column is None:
        raise ValueError(f"CosMx file is missing a {kind} column; expected one of {', '.join(candidates)}.")
    return column


def _normalize_fov(value: Any) -> str:
    """Normalise FOV identifiers such as ``1``, ``1.0`` and ``001`` to ``'1'``."""
    return str(int(float(value)))


def _cell_index(cell_ids: pd.Series, fovs: pd.Series) -> pd.Index:
    """Build a unique cell index from vendor cell ID and FOV columns."""
    return pd.Index(cell_ids.astype(str).str.cat(fovs.map(_normalize_fov), sep="_"))
