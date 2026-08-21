from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array import Array
from dask_image.imread import imread
from scipy.sparse import csr_matrix
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Identity, Translation

from sparrow.image._image import add_image_layer, add_labels_layer
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

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_FOV_IN_FILENAME = re.compile(r"_F(\d+)", flags=re.IGNORECASE)

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

_BLOCKSIZE = "128MB"


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
    """Validated inputs and discovered FOVs for one CosMx dataset root."""

    path: Path
    transcripts_file: Path
    images_dir: Path
    fov_positions_file: Path | None
    labels_dir: Path | None
    counts_file: Path | None
    metadata_file: Path | None
    image_fov_files: tuple[tuple[str, Path], ...]
    label_fov_files: tuple[tuple[str, Path], ...]
    fov_origins: dict[str, tuple[float, float]] | None
    transcripts: dd.DataFrame


def cosmx(
    path: str | Path | list[str] | list[Path],
    to_coordinate_system: str | list[str] = "global",
    dataset_id: str | list[str] | None = None,
    keep_gene_names: str | Path | Iterable[str] | None = None,
    cells_labels: bool = False,
    cells_table: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    output: str | Path | None = None,
) -> SpatialData:
    """Read *CosMx* data into a ``SpatialData`` object.

    Images and transcripts are required. Vendor cell labels and the vendor
    cell-by-gene table are optional. Transcripts are read lazily into a single
    points layer using global pixel coordinates when those columns exist.
    Gene names can be filtered to a whitelist before any coordinate work is performed.
    Multi-FOV datasets require a valid ``fov_positions_file.csv`` with origins
    for every image and requested label FOV; single-FOV datasets may use the
    identity transformation when no positions file is available.

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
        delimited file whose first column contains the gene panel. A single
        gene name is not supported. String values are interpreted as panel file paths.
        If ``None``, no gene filtering is performed.
        Stored transcript layers use Sparrow's canonical ``gene`` column.
    cells_labels
        Whether to read vendor ``CellLabels`` masks. Automatically enabled when
        ``cells_table`` is ``True`` because the cell table is linked to the vendor label masks.
    cells_table
        Whether to read the vendor ``exprMat`` counts and cell metadata table.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments forwarded to the image and labels models. Supported
        keys are ``chunks`` and ``scale_factors``.
    output
        Path where the resulting ``SpatialData`` object is backed. If ``None``,
        the result is returned in memory.

    Returns
    -------
    SpatialData
        CosMx images as ``{fov}_image_{coordinate_system}`` and transcripts as
        ``transcripts_{coordinate_system}``. Optional labels and tables use
        ``{fov}_labels_{coordinate_system}`` and ``table_{coordinate_system}``.

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

    # Warn about model options that this reader cannot forward to its image and label models.
    unsupported_image_model_keys = [
        str(key) for key in image_models_kwargs if key not in {"chunks", "scale_factors"}
    ]
    if unsupported_image_model_keys:
        log.warning(
            "Ignoring unsupported 'image_models_kwargs' keys: %s. Supported keys are 'chunks' and 'scale_factors'.",
            ", ".join(sorted(unsupported_image_model_keys)),
        )

    # Vendor tables annotate label layers, so reading the table implies reading labels.
    if cells_table and not cells_labels:
        log.info("Setting 'cells_labels' to True so the vendor table can annotate CosMx label layers.")
        cells_labels = True

    log.info(
        "Starting CosMx read for %d dataset(s); cells_labels=%s; cells_table=%s; coordinate systems=%s.",
        len(dataset_ids),
        cells_labels,
        cells_table,
        coordinate_systems,
    )

    # Load the optional keep list once so it is shared by all datasets.
    keep_genes = _load_keep_gene_names(keep_gene_names)

    if keep_genes is not None:
        log.info("Keeping %d CosMx genes while reading transcripts.", len(keep_genes))
    else:
        log.info("No CosMx gene whitelist supplied; no gene filtering will be applied.")

    # Validate every dataset before creating an output store so invalid later datasets cannot leave partial output.
    dataset_specs: list[_CosmxDataset] = []
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
        dataset_specs.append(
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
        log.info("Creating backed CosMx output at '%s'.", output)
        sdata.write(output)
        sdata = read_zarr(output)

    for dataset_index, (dataset, coordinate_system) in enumerate(
        zip(dataset_specs, coordinate_systems, strict=True),
        start=1,
    ):
        log.info(
            "Reading CosMx dataset %d/%d from '%s' into coordinate system '%s'.",
            dataset_index,
            len(dataset_specs),
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
            imread_kwargs=imread_kwargs,
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
    """Discover and validate one CosMx dataset before any layers are created."""
    # Find all relevant dataset_files for this dataset
    dataset_files = _discover_files(path, dataset_id=dataset_id, cells_labels=cells_labels, cells_table=cells_table)

    # Validate that every FOV in the images, labels, and transcripts has a known origin.
    image_fov_files = tuple(_fov_files(dataset_files.images_dir))
    if not image_fov_files:
        raise FileNotFoundError(f"No CosMx FOV images found in '{dataset_files.images_dir}'.")

    if cells_labels:
        if dataset_files.labels_dir is None:
            raise FileNotFoundError(f"CosMx labels directory not found under '{path}'.")
        label_fov_files = tuple(_fov_files(dataset_files.labels_dir))
        if not label_fov_files:
            raise FileNotFoundError(f"No CosMx FOV labels found in '{dataset_files.labels_dir}'.")
    else:
        label_fov_files = ()

    # FOV origins are translations only; they are never estimated from cell centroids.
    fov_origins = _load_fov_origins(dataset_files.fov_positions)

    # Read and gene-filter transcripts once so preflight and layer creation share one lazy graph.
    transcripts = _read_transcript_table(dataset_files.transcripts)
    gene_column = _require_column(transcripts.columns, _GENE_COLUMNS, kind="gene")
    if keep_genes is not None:
        transcripts = transcripts[transcripts[gene_column].isin(list(keep_genes))]

    # Include local transcript FOVs in the geometry check, even when images contain one FOV only.
    transcript_fovs = _get_transcript_fovs(transcripts, transcripts_path=dataset_files.transcripts)
    required_fovs = {fov for fov, _ in image_fov_files}
    required_fovs.update(fov for fov, _ in label_fov_files)
    required_fovs.update(transcript_fovs)
    _validate_fov_origins(
        required_fovs=required_fovs,
        fov_origins=fov_origins,
        fov_positions_path=dataset_files.fov_positions,
    )

    # Normalize and validate transcript coordinates before any output store or image layer is created.
    transcripts = _prepare_transcripts(
        transcripts=transcripts,
        transcripts_path=dataset_files.transcripts,
        gene_column=gene_column,
        fov_origins=fov_origins,
        require_fov_origins=len(required_fovs) > 1,
    )

    return _CosmxDataset(
        path=path,
        transcripts_file=dataset_files.transcripts,
        images_dir=dataset_files.images_dir,
        fov_positions_file=dataset_files.fov_positions,
        labels_dir=dataset_files.labels_dir,
        counts_file=dataset_files.counts,
        metadata_file=dataset_files.metadata,
        image_fov_files=image_fov_files,
        label_fov_files=label_fov_files,
        fov_origins=fov_origins,
        transcripts=transcripts,
    )


def _add_dataset(
    sdata: SpatialData,
    dataset: _CosmxDataset,
    coordinate_system: str,
    keep_genes: set[str] | None,
    cells_labels: bool,
    cells_table: bool,
    imread_kwargs: Mapping[str, Any],
    image_models_kwargs: Mapping[str, Any],
) -> SpatialData:
    """Add one validated CosMx dataset to ``sdata``."""
    sdata = _add_images(
        sdata,
        images_dir=dataset.images_dir,
        fov_files=dataset.image_fov_files,
        coordinate_system=coordinate_system,
        fov_origins=dataset.fov_origins,
        imread_kwargs=imread_kwargs,
        image_models_kwargs=image_models_kwargs,
    )

    sdata = _add_transcripts(
        sdata,
        transcripts=dataset.transcripts,
        transcripts_path=dataset.transcripts_file,
        coordinate_system=coordinate_system,
    )

    if cells_labels:
        sdata = _add_labels(
            sdata,
            labels_dir=dataset.labels_dir,
            fov_files=dataset.label_fov_files,
            coordinate_system=coordinate_system,
            fov_origins=dataset.fov_origins,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
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
    """Locate the standard CosMx files under ``path``."""
    # Check for the required transcript file through the known suffixes, preferring parquet over CSV when both exist.
    transcripts = _find_suffix_file(path, dataset_id, _TRANSCRIPT_SUFFIXES)
    if transcripts is None:
        raise FileNotFoundError(
            f"CosMx transcript file not found in '{path}'. Expected a file ending with {', '.join(_TRANSCRIPT_SUFFIXES)}."
        )

    # Check for the required image directory through the known candidates, or any FOV-containing subdirectory.
    images_dir = _find_directory(path, _IMAGE_DIR_CANDIDATES, skip_names={_LABEL_DIR_NAME})
    if images_dir is None:
        raise FileNotFoundError(
            f"CosMx image directory not found in '{path}'. Expected one of {', '.join(_IMAGE_DIR_CANDIDATES)}."
        )

    # Resolve and validate the labels directory only when labels were requested.
    labels_dir: Path | None = None
    if cells_labels:
        labels_dir = path / _LABEL_DIR_NAME
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"CosMx labels directory not found in '{path}'. Expected '{_LABEL_DIR_NAME}'.")
    
    # Infer one dataset identifier from the required transcript file so related files cannot come from another dataset.
    if dataset_id is None:
        current_transcript_suffix = next(
            suffix for suffix in _TRANSCRIPT_SUFFIXES if transcripts.name.endswith(suffix)
        )
        # Derive the shared dataset prefix from the transcript filename selected above.
        resolved_dataset_id = transcripts.name[: -len(current_transcript_suffix)]
    else:
        resolved_dataset_id = dataset_id

    # Find the optional FOV positions file, counts file, and metadata file through their known suffixes.
    fov_positions = _find_suffix_file(path, resolved_dataset_id, _FOV_POSITIONS_SUFFIXES)
    counts = _find_suffix_file(path, resolved_dataset_id, _COUNTS_SUFFIXES)
    metadata = _find_suffix_file(path, resolved_dataset_id, _METADATA_SUFFIXES)
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


def _add_images(
    sdata: SpatialData,
    images_dir: Path,
    fov_files: tuple[tuple[str, Path], ...],
    coordinate_system: str,
    fov_origins: Mapping[str, tuple[float, float]] | None,
    imread_kwargs: Mapping[str, Any],
    image_models_kwargs: Mapping[str, Any],
) -> SpatialData:
    """Add one image layer per FOV without stitching."""
    if not fov_files:
        raise FileNotFoundError(f"No CosMx FOV images found in '{images_dir}'.")

    chunks = image_models_kwargs.get("chunks")
    scale_factors = image_models_kwargs.get("scale_factors")

    for fov, image_path in fov_files:
        log.info("Reading CosMx image for FOV %s from '%s'.", fov, image_path)

        # Keep the image lazy; dask_image only builds a graph here.
        image = _normalize_image_dims(imread(image_path, **imread_kwargs), target_dims="cyx")
        sdata = add_image_layer(
            sdata,
            arr=image,
            output_layer=f"{fov}_image_{coordinate_system}",
            dims=("c", "y", "x"),
            chunks=chunks,
            transformations={coordinate_system: _fov_translation(fov, fov_origins)},
            scale_factors=scale_factors,
            overwrite=False,
        )

    return sdata


def _add_labels(
    sdata: SpatialData,
    labels_dir: Path,
    fov_files: tuple[tuple[str, Path], ...],
    coordinate_system: str,
    fov_origins: Mapping[str, tuple[float, float]] | None,
    imread_kwargs: Mapping[str, Any],
    image_models_kwargs: Mapping[str, Any],
) -> SpatialData:
    """Add one vendor labels layer per FOV."""
    if not fov_files:
        raise FileNotFoundError(f"No CosMx FOV labels found in '{labels_dir}'.")

    chunks = image_models_kwargs.get("chunks")
    scale_factors = image_models_kwargs.get("scale_factors")

    for fov, labels_path in fov_files:
        log.info("Reading CosMx labels for FOV %s from '%s'.", fov, labels_path)

        # Vendor masks are integer label images in FOV-local pixel coordinates.
        labels = _normalize_image_dims(imread(labels_path, **imread_kwargs), target_dims="yx").astype(np.uint32)
        sdata = add_labels_layer(
            sdata,
            arr=labels,
            output_layer=f"{fov}_labels_{coordinate_system}",
            dims=("y", "x"),
            chunks=chunks,
            transformations={coordinate_system: _fov_translation(fov, fov_origins)},
            scale_factors=scale_factors,
            overwrite=False,
        )

    return sdata


def _prepare_transcripts(
    transcripts: dd.DataFrame,
    transcripts_path: Path,
    gene_column: str,
    fov_origins: Mapping[str, tuple[float, float]] | None,
    require_fov_origins: bool,
) -> dd.DataFrame:
    """Normalize, place, and validate CosMx transcript coordinates."""
    global_x = _optional_column(transcripts.columns, _GLOBAL_X_COLUMNS)
    global_y = _optional_column(transcripts.columns, _GLOBAL_Y_COLUMNS)

    if global_x is not None and global_y is not None:
        log.info("Using CosMx global pixel columns '%s' and '%s' for transcript coordinates.", global_x, global_y)
        transcripts = transcripts.rename(columns={gene_column: _GENES_KEY, global_x: "x", global_y: "y"})
    else:
        local_x = _require_column(transcripts.columns, _LOCAL_X_COLUMNS, kind="local x")
        local_y = _require_column(transcripts.columns, _LOCAL_Y_COLUMNS, kind="local y")
        fov_column = _optional_column(transcripts.columns, _FOV_COLUMNS)

        # Apply FOV origins to local coordinates when the transcript schema supports that placement.
        if fov_origins is not None and fov_column is not None:
            log.info("Global transcript coordinates are absent; applying FOV translations from fov_positions.")
            transcripts = transcripts.map_partitions(
                _apply_fov_origins,
                gene_column=gene_column,
                local_x=local_x,
                local_y=local_y,
                fov_column=fov_column,
                fov_origins=fov_origins,
                meta=pd.DataFrame(
                    {
                        _GENES_KEY: pd.Series(dtype="object"),
                        "x": pd.Series(dtype="float64"),
                        "y": pd.Series(dtype="float64"),
                    }
                ),
            )
        elif require_fov_origins:
            raise ValueError(
                f"CosMx transcript file '{transcripts_path}' uses local pixel coordinates, but complete FOV "
                "origins are unavailable. Provide global pixel columns or a valid FOV positions file."
            )
        else:
            log.warning(
                "CosMx transcript file '%s' has no usable global pixel coordinates or FOV placement metadata; "
                "using local pixel coordinates as-is.",
                transcripts_path,
            )
            transcripts = transcripts.rename(columns={gene_column: _GENES_KEY, local_x: "x", local_y: "y"})

    # Keep only the columns Sparrow allocation needs so extra CosMx fields are not persisted.
    transcripts = transcripts[[_GENES_KEY, "x", "y"]]

    # Convert transcript coordinates to numeric values before checking their geometry.
    transcripts["x"] = transcripts["x"].astype(float)
    transcripts["y"] = transcripts["y"].astype(float)

    # Reject non-finite coordinates before any output store or points layer is created.
    _validate_transcript_coordinates(transcripts, transcripts_path=transcripts_path)

    return transcripts


def _add_transcripts(
    sdata: SpatialData,
    transcripts: dd.DataFrame,
    transcripts_path: Path,
    coordinate_system: str,
) -> SpatialData:
    """Read CosMx transcripts lazily into one identity-transformed points layer."""
    log.info("Reading CosMx transcripts from '%s'.", transcripts_path)

    return add_points_layer(
        sdata,
        ddf=transcripts,
        output_layer=f"transcripts_{coordinate_system}",
        coordinates={"x": "x", "y": "y"},
        transformations={coordinate_system: Identity()},
        overwrite=False,
    )


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

    # Point the table at the renamed per-FOV labels layers stored in this coordinate system.
    metadata[_REGION_KEY] = pd.Categorical(
        metadata[metadata_fov].map(lambda fov: f"{_normalize_fov(fov)}_labels_{coordinate_system}")
    )
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
    """Return a named image folder, or a FOV-containing image subdirectory."""
    for name in candidates:
        candidate = path / name
        if candidate.is_dir():
            return candidate

    for child in sorted(path.iterdir()):
        if child.is_dir() and child.name not in skip_names and _fov_files(child):
            return child

    return None


def _fov_files(directory: Path) -> list[tuple[str, Path]]:
    """Return ``(fov, path)`` pairs for image-like files whose names contain ``_F<number>``."""
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for child in sorted(directory.iterdir()):
        if not child.is_file() or child.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        match = _FOV_IN_FILENAME.search(child.name)
        if match is None:
            continue

        # Normalise F001 and F1 to the same FOV identifier.
        fov = str(int(match.group(1)))
        if fov in seen:
            log.warning("Skipping extra CosMx file '%s' for FOV %s.", child, fov)
            continue

        seen.add(fov)
        files.append((fov, child))

    return files


def _load_fov_origins(fov_positions_path: Path | None) -> dict[str, tuple[float, float]] | None:
    """Read FOV translations from ``fov_positions_file.csv`` when pixel columns exist."""
    if fov_positions_path is None:
        return None

    positions = pd.read_csv(fov_positions_path, header=0)
    fov_column = _optional_column(positions.columns, _FOV_COLUMNS)
    x_column = _optional_column(positions.columns, _FOV_X_COLUMNS)
    y_column = _optional_column(positions.columns, _FOV_Y_COLUMNS)

    if fov_column is None or x_column is None or y_column is None:
        log.warning(
            "CosMx FOV positions file '%s' does not contain pixel origin columns; "
            "FOV images will use an identity transformation.",
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


def _get_transcript_fovs(transcripts: dd.DataFrame, transcripts_path: Path) -> frozenset[str]:
    """Return FOV identifiers needed to place local transcript coordinates."""
    global_x = _optional_column(transcripts.columns, _GLOBAL_X_COLUMNS)
    global_y = _optional_column(transcripts.columns, _GLOBAL_Y_COLUMNS)

    # Global transcript coordinates already have a common placement and need no FOV-origin lookup.
    if global_x is not None and global_y is not None:
        return frozenset()

    # Validate local coordinate columns while the transcript schema is available during preflight.
    _require_column(transcripts.columns, _LOCAL_X_COLUMNS, kind="local x")
    _require_column(transcripts.columns, _LOCAL_Y_COLUMNS, kind="local y")
    fov_column = _optional_column(transcripts.columns, _FOV_COLUMNS)
    if fov_column is None:
        return frozenset()

    # Compute only distinct FOV identifiers before any images or output stores are created.
    transcript_fovs = transcripts[fov_column].drop_duplicates().compute()
    try:
        return frozenset(_normalize_fov(fov) for fov in transcript_fovs)
    except ValueError as error:
        raise ValueError(
            f"CosMx transcript file '{transcripts_path}' contains an invalid FOV identifier."
        ) from error


def _validate_transcript_coordinates(transcripts: dd.DataFrame, transcripts_path: Path) -> None:
    """Reject non-numeric or non-finite transcript coordinates."""
    try:
        # Check coordinates across partitions lazily and compute reduction
        has_invalid = ~(
            (transcripts["x"].map_partitions(np.isfinite, meta=pd.Series(dtype="bool")))
            & (transcripts["y"].map_partitions(np.isfinite, meta=pd.Series(dtype="bool")))
        ).all().compute()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"CosMx transcript file '{transcripts_path}' contains non-numeric coordinate values."
        ) from error

    # Stop before layer creation when any coordinate is NaN, positive infinity, or negative infinity.
    if bool(has_invalid):
        raise ValueError(f"CosMx transcript file '{transcripts_path}' contains non-finite coordinates.")


def _validate_fov_origins(
    required_fovs: Iterable[str],
    fov_origins: Mapping[str, tuple[float, float]] | None,
    fov_positions_path: Path | None,
) -> None:
    """Require complete FOV origins when more than one FOV is being loaded."""
    # Deduplicate FOV identifiers because an image and its label mask share one origin.
    required_fovs = set(required_fovs)

    # A single FOV can use identity coordinates when no placement metadata is available.
    if len(required_fovs) <= 1:
        return

    # Reject absent or structurally invalid positions files before reading any image data.
    if fov_origins is None:
        if fov_positions_path is None:
            positions_description = "no FOV positions file was found"
        else:
            positions_description = f"FOV positions file '{fov_positions_path}' has no usable pixel origins"
        raise ValueError(
            f"CosMx dataset contains multiple FOVs ({', '.join(sorted(required_fovs))}), but "
            f"{positions_description}. Valid FOV origins are required for multi-FOV analysis."
        )

    # Reject a positions file that does not describe every FOV that will be loaded.
    missing_fovs = sorted(required_fovs.difference(fov_origins))
    if missing_fovs:
        raise ValueError(
            "CosMx FOV positions metadata is incomplete; missing origins for FOVs "
            f"{', '.join(missing_fovs)}. Valid origins are required for every multi-FOV image and label layer."
        )


def _fov_translation(
    fov: str,
    fov_origins: Mapping[str, tuple[float, float]] | None,
) -> Identity | Translation:
    """Return the translation that places one FOV image into the global pixel system."""
    if fov_origins is None:
        return Identity()

    # Treat a missing origin as invalid when a positions mapping was supplied.
    try:
        origin = fov_origins[fov]
    except KeyError as error:
        raise ValueError(
            f"CosMx FOV '{fov}' is missing from the FOV positions metadata; "
            "an origin is required to place this FOV."
        ) from error

    # Reject values that cannot represent the two-dimensional FOV origin used by CosMx.
    if not isinstance(origin, tuple) or len(origin) != 2:
        raise ValueError(
            f"CosMx FOV '{fov}' received unsupported transformation type "
            f"'{type(origin).__name__}'. Sparrow supports only numeric two-dimensional origins here."
        )

    # Convert the numeric FOV origin into the translation used by image and label layers.
    return Translation(list(origin), axes=("x", "y"))


def _read_transcript_table(path: Path) -> dd.DataFrame:
    """Read CosMx transcripts as a Dask dataframe from parquet or CSV."""
    if path.suffix.lower() == ".parquet":
        return dd.read_parquet(path)

    return dd.read_csv(path, header=0, blocksize=_BLOCKSIZE)


def _apply_fov_origins(
    partition: pd.DataFrame,
    gene_column: str,
    local_x: str,
    local_y: str,
    fov_column: str,
    fov_origins: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Add FOV origin translations to local transcript coordinates."""
    # Pre-build coordinate lookup maps to avoid row-by-row lambda evaluation in Pandas
    x_map = {fov: origin[0] for fov, origin in fov_origins.items()}
    y_map = {fov: origin[1] for fov, origin in fov_origins.items()}

    # Normalize FOVs and map to numeric origin offsets vectorially
    fov_ids = partition[fov_column].map(_normalize_fov)
    origin_x = fov_ids.map(x_map)
    origin_y = fov_ids.map(y_map)

    return pd.DataFrame(
        {
            _GENES_KEY: partition[gene_column].astype(str),
            "x": partition[local_x].astype(float) + origin_x.astype(float),
            "y": partition[local_y].astype(float) + origin_y.astype(float),
        }
    )


def _normalize_image_dims(array: Array, target_dims: str = "cyx") -> Array:
    """Normalise an image or label array to ``(c, y, x)`` or ``(y, x)`` without flipping axes."""
    array = da.squeeze(array)

    if target_dims == "cyx":
        if array.ndim == 2:
            return array[None, ...]
        if array.ndim == 3:
            # RGB/RGBA FOV composites are stored as (y, x, c).
            if array.shape[-1] in (3, 4) and array.shape[0] not in (3, 4):
                return da.moveaxis(array, -1, 0)
            return array
    elif target_dims == "yx":
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            # Drop a trailing or leading singleton channel if the vendor wrote RGB-like labels.
            if array.shape[0] == 1:
                return array[0]
            if array.shape[-1] == 1:
                return array[..., 0]

    raise ValueError(f"CosMx image has unsupported shape {array.shape}; expected 2D or 3D for target '{target_dims}'.")


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
