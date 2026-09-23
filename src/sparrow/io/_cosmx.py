from __future__ import annotations

import csv
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
from scipy.sparse import csr_matrix
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Identity
from spatialdata.transformations._utils import _set_transformations, compute_coordinates
from xarray import DataTree

from sparrow.image._manager import ImageLayerManager, LabelLayerManager
from sparrow.points._points import add_points_layer
from sparrow.table._table import add_table_layer
from sparrow.utils._keys import _CELL_INDEX, _GENES_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
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

# Obs column that keeps the vendor's per-FOV cell number once _INSTANCE_KEY (= "cell_ID")
# has been repurposed to hold the dataset-global CellLabels mask value.
_VENDOR_CELL_ID_COLUMN = "fov_cell_ID"

_BLOCKSIZE = "256MB"

# Bytes read from the end of a vendor CSV to find its last line.
# The window doubles until it holds that whole line.
_CSV_TAIL_BYTES = 65_536
_CSV_TAIL_MAX_BYTES = 16 * 1024 * 1024

# Dtypes used to accumulate the vendor counts CSV into a sparse matrix, plus the byte budget for
# one streamed chunk of it. Chunk rows are derived from that budget, since a chunk is dense.
_COUNTS_VALUE_DTYPE = np.int32
_COUNTS_INDEX_DTYPE = np.int32
_COUNTS_CHUNK_BYTES = 512_000_000


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
class _FovPositions:
    """FOV placement read from ``fov_positions_file.csv``, in the vendor's global pixel system."""

    # Per-FOV tile origin, as (x, y). The vendor's y is the tile's *top* edge, because its global
    # pixel y axis increases upward while raster rows increase downward.
    origins: dict[int, tuple[float, float]]
    # Global pixel y of the mosaic's first raster row, i.e. the top edge of the topmost tile.
    top_global_y: float


@dataclass(frozen=True)
class _CosmxDataset:
    """One validated CosMx dataset root: its resolved files plus its lazy transcript frame."""

    path: Path
    files: _CosmxFiles
    transcripts: dd.DataFrame
    top_global_y: float | None


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

    All layers are expressed in the mosaic's raster pixel space. The vendor's
    global pixel y axis increases upward while raster rows increase downward, so
    transcript and cell-centre y coordinates are converted using the mosaic's top
    edge, taken from ``fov_positions_file.csv``. That file is therefore needed for
    points and rasters to overlay, in addition to placing local FOV coordinates.
    A vendor CSV file whose last line is incomplete, as an interrupted download
    or copy leaves it, is rejected before any output is written. Transcripts with
    a non-finite coordinate are rejected as each partition is read.

    When the vendor table is read, its instance key holds the integer value stored
    in the ``CellLabels`` mask, reconstructed from ``fov`` and ``cell_ID``, so the
    table can be joined against its labels layer. The vendor's per-FOV cell number
    is preserved in the ``fov_cell_ID`` observation column.

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
        duplicated, ``keep_gene_names`` is a scalar gene name or empty whitelist,
        a vendor CSV file is truncated, or a transcript has a non-finite coordinate.
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
            ", ".join(sorted(unsupported_image_model_keys)),
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
        "Finished CosMx read: %d image(s), %d label(s), %d point layer(s), and %d table(s) "
        "were added to the spatial data object.",
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

    # Reject a vendor CSV that was cut off mid-line before any of it is parsed.
    # pandas would pad its incomplete last line with NaN instead of raising
    csv_files: list[tuple[Path | None, str]] = [
        (dataset_files.transcripts, "transcript"),
        (dataset_files.fov_positions, "FOV positions"),
    ]
    # The counts and metadata files are only parsed when the vendor table is requested.
    if cells_table:
        csv_files += [(dataset_files.counts, "counts"), (dataset_files.metadata, "metadata")]
    for csv_path, kind in csv_files:
        # Parquet transcripts and an absent positions file leave no CSV to check.
        if csv_path is not None and csv_path.suffix.lower() == ".csv":
            _reject_truncated_csv(csv_path, kind=kind)

    # Load FOV placement from the optional FOV positions file. It supplies both the per-FOV tile
    # origins and the mosaic's top edge, which is what maps vendor global pixels onto raster rows.
    fov_positions = _load_fov_positions(dataset_files.fov_positions)

    # Read transcripts lazily from parquet or CSV.
    if dataset_files.transcripts.suffix.lower() == ".parquet":
        transcripts = dd.read_parquet(dataset_files.transcripts)
    else:
        # First, read only the header row so the dtype of every column can be explicitly specified
        # before Dask ever samples the file itself.
        # This avoids Dask's unreliable dtype inference for large multi-gigabyte CSVs.
        header_columns = pd.read_csv(dataset_files.transcripts, header=0, nrows=0).columns
        transcripts = dd.read_csv(
            dataset_files.transcripts,
            header=0,
            blocksize=_BLOCKSIZE,
            dtype=_resolve_transcript_csv_dtypes(header_columns),
        )

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
        fov_positions=fov_positions,
        fov_positions_path=dataset_files.fov_positions,
    )

    return _CosmxDataset(
        path=path,
        files=dataset_files,
        transcripts=transcripts,
        top_global_y=None if fov_positions is None else fov_positions.top_global_y,
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
        dataset.files.images_dir,
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
        # Guaranteed by _discover_files when labels are requested.
        assert dataset.files.labels_dir is not None
        # Process multiscale integer label masks and register with SpatialData.
        labels_tree = _process_multiscale_raster(
            dataset.files.labels_dir,
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
        # Guaranteed by _discover_files when the table is requested.
        assert dataset.files.counts is not None and dataset.files.metadata is not None
        sdata = _add_table(
            sdata,
            counts_path=dataset.files.counts,
            metadata_path=dataset.files.metadata,
            coordinate_system=coordinate_system,
            keep_genes=keep_genes,
            top_global_y=dataset.top_global_y,
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


def _reject_truncated_csv(path: Path, kind: str) -> None:
    """Raise when a vendor CSV ends in an incomplete line, i.e. it was cut off while being written or copied.

    pandas pads a line with fewer fields than the header with NaN instead of raising. A truncated
    file therefore otherwise surfaces only once the output store is being written, as a misleading
    symptom such as a non-finite coordinate, or not at all when the cut leaves a finite but wrong
    value (``51054`` cut to ``510``). Only the header and the last line are read, so the check is
    cheap for a file of any size.
    """
    # The header defines how many fields every complete line must have. The read is capped, so a
    # file without any line break is never read whole.
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        n_header_fields = _count_csv_fields(handle.readline(_CSV_TAIL_MAX_BYTES))
    # An empty or unparseable header leaves nothing to compare against; the regular parsing reports it.
    if n_header_fields == 0:
        return

    last_line, ends_with_line_break = _read_last_line(path)
    n_fields = _count_csv_fields(last_line)

    # A cut can only remove fields, so fewer fields than the header means the file ends mid-line.
    # More fields cannot come from a cut and are left to pandas.
    if n_fields < n_header_fields:
        line_break_note = "" if ends_with_line_break else " and the file does not end with a line break"
        raise ValueError(
            f"CosMx {kind} file '{path}' appears to be truncated: its last line has {n_fields} of the "
            f"header's {n_header_fields} fields ({last_line[:80]!r}){line_break_note}. The file was "
            "probably not completely written or copied; re-export or re-download it."
        )

    # A completed CSV write, ends in a line break, so a missing one hints at a cut that fell inside the
    # last field. The line has every field, though, and some tools omit the break, so only warn.
    if not ends_with_line_break:
        log.warning(
            "CosMx %s file '%s' does not end with a line break, so it may have been cut off while being "
            "written or copied. Its last line has every field, so reading continues.",
            kind,
            path,
        )


def _read_last_line(path: Path) -> tuple[str, bool]:
    """Return the last non-empty line of a text file, and whether the file ends with a line break.

    A last line longer than ``_CSV_TAIL_MAX_BYTES`` is returned cut to its final bytes.
    """
    size = path.stat().st_size
    window = _CSV_TAIL_BYTES
    with path.open("rb") as handle:
        while True:
            start = max(size - window, 0)
            handle.seek(start)
            tail = handle.read()
            # Trailing line breaks, blank lines included, do not form a last line of their own.
            content = tail.rstrip(b"\r\n")
            # Stop the loop when the remaining content contains a newline,
            # meaning the start of the final line is within the window,
            # or once the window covers the whole file or has reached its cap.
            if b"\n" in content or start == 0 or window >= _CSV_TAIL_MAX_BYTES:
                break
            window *= 2

    # Keep only what follows the last line break, dropping a Windows carriage return.
    last_line = content.rsplit(b"\n", 1)[-1].rstrip(b"\r")
    return last_line.decode("utf-8", errors="replace"), tail.endswith(b"\n")


def _count_csv_fields(line: str) -> int:
    """Count the fields of one CSV line, or return 0 when it cannot be parsed as CSV."""
    # The csv module honours quoting, so a quoted comma is not miscounted as a separator.
    try:
        return len(next(csv.reader([line]), []))
    except csv.Error:
        # csv rejects a single field longer than its field size limit, such as a long run of zero padding.
        return 0


def _resolve_transcript_csv_dtypes(columns: Iterable[str]) -> dict[str, Any]:
    """Build an explicit per-column dtype map for a CosMx transcripts CSV.

    Dask infers each column's dtype from a small sample of the file (256KB by default), which is
    unreliable for a multi-gigabyte CosMx export. A column that is blank in that sample defaults
    to float64 (an all-NaN column looks numeric) but can hold real text further into the file.
    Such a mismatch crashes Dask's cross-partition dtype reconciliation once the later partitions
    are actually read. Every column this reader gives a specific meaning to is assigned its
    correct dtype explicitly here instead, so no sampling is involved for it.
    Every other (vendor passthrough) column is read as a string, since this reader never depends
    on a passthrough column's dtype.
    """
    dtype: dict[str, Any] = {}

    # Gene/target values are never numeric.
    gene_column = _optional_column(columns, _GENE_COLUMNS)
    if gene_column is not None:
        dtype[gene_column] = str

    # Pixel coordinates are cast to float wherever they are used (see below), so declaring every
    # candidate column float64 up front avoids Dask ever mistaking one for a clean integer column.
    for candidates in (_GLOBAL_X_COLUMNS, _GLOBAL_Y_COLUMNS, _LOCAL_X_COLUMNS, _LOCAL_Y_COLUMNS):
        column = _optional_column(columns, candidates)
        if column is not None:
            dtype[column] = np.float64

    # CosMx FOV and cell IDs are always populated whole numbers.
    for candidates in (_FOV_COLUMNS, _CELL_ID_COLUMNS):
        column = _optional_column(columns, candidates)
        if column is not None:
            dtype[column] = np.int64

    # Any remaining column is a vendor passthrough column this reader never inspects.
    for column in columns:
        # If key is not already in the dictionary, insert it with value.
        # Otherwise, leave the existing value unchanged.
        dtype.setdefault(column, str)

    return dtype


def _prepare_transcripts(
    transcripts: dd.DataFrame,
    transcripts_path: Path,
    gene_column: str,
    fov_positions: _FovPositions | None,
    fov_positions_path: Path | None,
) -> dd.DataFrame:
    """Normalize, place, and validate CosMx transcript coordinates in raster pixel space.

    Vendor global pixel coordinates are not raster coordinates: their y axis increases *upward*,
    while the image and label mosaics are stored with rows increasing downward. The vendor image
    and label pyramids are registered with an identity transformation, so transcripts must be
    expressed in that same raster space or they land mirrored across the mosaic.
    """
    # Check if global pixel coordinates are present in the transcript file.
    global_x = _optional_column(transcripts.columns, _GLOBAL_X_COLUMNS)
    global_y = _optional_column(transcripts.columns, _GLOBAL_Y_COLUMNS)

    if global_x is not None or global_y is not None:
        global_x = _require_column(transcripts.columns, _GLOBAL_X_COLUMNS, kind="global x")
        global_y = _require_column(transcripts.columns, _GLOBAL_Y_COLUMNS, kind="global y")
        log.info("Using CosMx global pixel columns '%s' and '%s' for transcript coordinates.", global_x, global_y)

        if fov_positions is None:
            # Without the positions file the mosaic's top edge is unknown, so the upward-increasing
            # vendor y cannot be converted into raster rows.
            # As a result, the points will not overlay the image and label layers.
            log.warning(
                "No usable CosMx FOV positions were found next to '%s', so global transcript y "
                "coordinates cannot be converted into raster rows! Transcripts will be vertically "
                "mirrored with respect to the image and label layers!",
                transcripts_path,
            )
            top_global_y = None
        else:
            log.info(
                "Converting CosMx global pixel y into raster rows using mosaic top edge y=%s.",
                fov_positions.top_global_y,
            )
            top_global_y = fov_positions.top_global_y

        # Place and validate each partition as it is read
        return transcripts.map_partitions(
            _apply_global_coordinates,
            gene_column=gene_column,
            global_x=global_x,
            global_y=global_y,
            top_global_y=top_global_y,
            transcripts_path=transcripts_path,
            meta=_canonical_coordinate_meta(transcripts, gene_column, global_x, global_y),
        )

    # Validate required local coordinate columns and FOV column when global coordinates are absent.
    local_x = _require_column(transcripts.columns, _LOCAL_X_COLUMNS, kind="local x")
    local_y = _require_column(transcripts.columns, _LOCAL_Y_COLUMNS, kind="local y")
    fov_column = _require_column(transcripts.columns, _FOV_COLUMNS, kind="fov")

    # Local coordinates are unplaceable without per-FOV origins to translate them by.
    if fov_positions is None:
        if fov_positions_path is None:
            positions_description = "no FOV positions file was found"
        else:
            positions_description = f"FOV positions file '{fov_positions_path}' has no usable pixel origins"
        raise ValueError(
            f"CosMx transcript file '{transcripts_path}' contains local FOV coordinates, but "
            f"{positions_description}; global transcript coordinates cannot be determined."
        )

    log.info("Global transcript coordinates are absent; applying FOV translations from fov_positions.")

    # Local pixel coordinates already run in raster orientation (y downward from the tile's top-left
    # corner), so placing them needs a plain translation per FOV rather than the global-y flip:
    # the tile's first raster row is the mosaic top edge minus that tile's global origin y.
    origin_x = pd.Series({fov: origin[0] for fov, origin in fov_positions.origins.items()}, dtype=float)
    origin_row = pd.Series(
        {fov: fov_positions.top_global_y - origin[1] for fov, origin in fov_positions.origins.items()},
        dtype=float,
    )

    # Use Dask's map_partitions to apply FOV origins to local coordinates in a fully vectorized manner.
    return transcripts.map_partitions(
        _apply_fov_origins,
        gene_column=gene_column,
        local_x=local_x,
        local_y=local_y,
        fov_column=fov_column,
        origin_x=origin_x,
        origin_row=origin_row,
        transcripts_path=transcripts_path,
        meta=_canonical_coordinate_meta(transcripts, gene_column, local_x, local_y),
    )


def _canonical_coordinate_meta(
    transcripts: dd.DataFrame,
    gene_column: str,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    """Build the meta frame describing a placed partition, for either coordinate branch."""
    # Dask builds a task graph per partition, for which it needs to know the columns and dtypes the
    # partition function returns. Both branches return the same shape: vendor columns untouched,
    # with the gene and coordinate columns renamed to Sparrow's canonical names and cast to float.
    meta = transcripts._meta.rename(columns={gene_column: _GENES_KEY, x_column: "x", y_column: "y"})
    meta["x"] = meta["x"].astype(float)
    meta["y"] = meta["y"].astype(float)
    return meta


def _add_table(
    sdata: SpatialData,
    counts_path: Path,
    metadata_path: Path,
    coordinate_system: str,
    keep_genes: set[str] | None,
    top_global_y: float | None,
) -> SpatialData:
    """Read the optional vendor cell-by-gene table, streaming counts to avoid a dense (cells x genes) read."""
    metadata = pd.read_csv(metadata_path, header=0)
    metadata_fov = _require_column(metadata.columns, _FOV_COLUMNS, kind="fov")
    metadata_cell = _require_column(metadata.columns, _CELL_ID_COLUMNS, kind="cell ID")

    # Some CosMx metadata exports carry more than one cell-ID alias at once (e.g. both 'cell_id'
    # and 'cell_ID', observed to hold different values).
    # SpatialData's table model rejects obs column names that are case-insensitive duplicates of each other,
    # so drop every alias that was not chosen as the canonical column above.
    redundant_cell_id_columns = [
        column for column in _CELL_ID_COLUMNS if column != metadata_cell and column in metadata.columns
    ]
    if redundant_cell_id_columns:
        log.info(
            "Dropping redundant CosMx cell ID column alias(es) %s from vendor metadata; using '%s'.",
            redundant_cell_id_columns,
            metadata_cell,
        )
        metadata = metadata.drop(columns=redundant_cell_id_columns)

    # The vendor's per-FOV cell ID restarts from 1 in every FOV, so it is only unique within a
    # single FOV, not across the dataset. Using it alone would silently merge unrelated cells
    # from different FOVs that happen to share the same local ID.
    # Building a dataset-global identifier by combining cell IDs with their corresponding FOVs.
    metadata = metadata.set_index(
        pd.Index(
            metadata[metadata_fov].astype(str) + "_" + metadata[metadata_cell].astype(str),
            name=_CELL_INDEX,
        )
    )

    # Build the dataset-global label ID for every vendor cell once, here. It is the value stored in
    # the CellLabels mask, it is what the counts rows are matched on below, and it becomes
    # _INSTANCE_KEY further down; deriving it once keeps those three uses from drifting apart.
    vendor_fovs = _normalize_integer_ids(metadata[metadata_fov]).to_numpy()
    vendor_cell_ids = _normalize_integer_ids(metadata[metadata_cell]).to_numpy()
    instance_keys = _vendor_label_ids(vendor_fovs, vendor_cell_ids)

    # Uniqueness has to hold for the paired integer key rather than only for its string form, both
    # because the key becomes the table's instance key and because the lookup below rejects a
    # non-unique index. The two are not the same test: '1' and '01' are distinct strings that
    # normalize to the same cell and would therefore pair to the same label.
    duplicated = pd.Index(instance_keys).duplicated()
    if duplicated.any():
        # Report the offending cells the way the vendor names them.
        # dict.fromkeys de-duplicates while preserving file order
        duplicate_keys = list(
            dict.fromkeys(
                f"{fov}_{cell}" for fov, cell in zip(vendor_fovs[duplicated], vendor_cell_ids[duplicated], strict=True)
            )
        )
        raise ValueError(
            f"CosMx metadata file '{metadata_path}' contains {len(duplicate_keys)} duplicate "
            f"fov+cell ID key(s), e.g. {', '.join(str(key) for key in duplicate_keys[:5])}. "
            "Every vendor cell must appear exactly once."
        )

    # Read only the header row so gene columns are known up front, before any counts rows are read.
    counts_columns = pd.read_csv(counts_path, header=0, nrows=0).columns
    counts_fov = _require_column(counts_columns, _FOV_COLUMNS, kind="fov")
    counts_cell = _require_column(counts_columns, _CELL_ID_COLUMNS, kind="cell ID")
    # Drop every identifier alias, not only the two chosen above: an export can carry more than one
    # alias at once (see the metadata handling above), and a leftover alias would otherwise be read
    # as a gene column and summed into the expression matrix as a cell count.
    identifier_columns = {column for column in (*_FOV_COLUMNS, *_CELL_ID_COLUMNS) if column in counts_columns}
    gene_columns = [column for column in counts_columns if column not in identifier_columns]
    if keep_genes is not None:
        gene_columns = [column for column in gene_columns if str(column) in keep_genes]

    # Log a memory estimate before the expensive read, so a too-large panel/cell count
    # is a visible warning up front rather than a silent multi-minute hang before an OOM.
    n_genes = len(gene_columns)
    # len(metadata) is an upper bound on the final cell count (counts is intersected against it below)
    estimated_dense_bytes = len(metadata) * n_genes * np.dtype(_COUNTS_VALUE_DTYPE).itemsize
    log.info(
        "CosMx vendor table has %d candidate cells (from metadata) and %d genes; a dense %s matrix "
        "would need up to ~%.2f GB of memory. Streaming '%s' into a sparse matrix instead.",
        len(metadata),
        n_genes,
        np.dtype(_COUNTS_VALUE_DTYPE),
        estimated_dense_bytes / 1e9,
        counts_path,
    )

    # Single pass over the counts file without materializing the full (cells x genes) matrix as a dense array
    X, seen = _read_counts_sparse(
        counts_path=counts_path,
        counts_fov=counts_fov,
        counts_cell=counts_cell,
        gene_columns=gene_columns,
        match_index=pd.Index(instance_keys),
    )

    # A cell belongs in the table only if it appears in both metadata and counts.
    # Filter the metadata to the cells that were actually seen in the counts file, preserving the original row order.
    # This later is necessary because AnnData aligns X, obs, and var positionally, not by index label.
    # Skip when every cell was seen (i.e all True mask) to avoid a copy of the full metadata frame.
    if not seen.all():
        metadata = metadata.loc[seen].copy()
        instance_keys = instance_keys[seen]

    # Point every vendor cell at the single global labels layer stored in this coordinate system.
    # from_codes builds the one-category column directly, without a len(metadata) Python list.
    metadata[_REGION_KEY] = pd.Categorical.from_codes(
        np.zeros(len(metadata), dtype=np.int8),
        categories=[f"labels_{coordinate_system}"],
        ordered=False,
    )

    # _INSTANCE_KEY is itself named "cell_ID", and it must hold the integer value stored in the
    # labels layer for the table to be joinable against it. Move the vendor's per-FOV cell number
    # aside first so that information is not lost when the instance key takes that name.
    metadata = metadata.rename(columns={metadata_cell: _VENDOR_CELL_ID_COLUMN})

    # TableLayerManager, which is called below, expects the _INSTANCE_KEY column to be present in
    # the obs metadata; it holds the label IDs derived from the vendor identifiers above.
    metadata[_INSTANCE_KEY] = instance_keys
    log.info(
        "Derived %d dataset-global CosMx cell label IDs from '%s' and '%s' for the '%s' table link.",
        len(metadata),
        metadata_fov,
        _VENDOR_CELL_ID_COLUMN,
        _INSTANCE_KEY,
    )

    # Create an AnnData object for the vendor table with a sparse expression matrix and the required obs/var structure.
    # Scipy's Compressed Sparse Row format is used because count matrices are typically very sparse,
    # so sparse storage saves substantial memory.
    adata = AnnData(
        X=X,
        obs=metadata,
        var=pd.DataFrame(index=pd.Index(gene_columns).astype(str)),
    )

    # Preserve global cell centres when the vendor metadata provides them.
    center_x = _optional_column(adata.obs.columns, _CENTER_X_GLOBAL_COLUMNS)
    center_y = _optional_column(adata.obs.columns, _CENTER_Y_GLOBAL_COLUMNS)
    # If they are present, copy them into the obsm spatial coordinates for Sparrow, following the conventional key
    # (x, y) pair used throughout the scanpy/squidpy/spatialdata ecosystem for per-cell spatial coordinates
    if center_x is not None and center_y is not None:
        centres = adata.obs[[center_x, center_y]].to_numpy(dtype=float)
        # These centres are vendor global pixels, so their y needs the same upward-to-downward
        # conversion applied to the transcripts; otherwise the table's coordinates would sit
        # mirrored against both the rasters and the points layer.
        if top_global_y is None:
            log.warning(
                "CosMx cell centres are stored in vendor global pixels because no FOV positions "
                "were available; they will not line up with the image and label layers."
            )
        else:
            centres[:, 1] = top_global_y - centres[:, 1]
        adata.obsm[_SPATIAL] = centres

    return add_table_layer(
        sdata,
        adata=adata,
        output_layer=f"table_{coordinate_system}",
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),  # a one-element list, ["labels_<coordinate_system>"]
        overwrite=False,
    )


def _read_counts_sparse(
    counts_path: Path,
    counts_fov: str,
    counts_cell: str,
    gene_columns: list[str],
    match_index: pd.Index,
) -> tuple[csr_matrix, np.ndarray]:
    """Stream the vendor counts CSV once and accumulate it directly into a sparse matrix.

    **Each counts row is placed at its cell's position in ``match_index``, matched on the paired
    integer label ID rebuilt from that row's own ``(fov, cell_ID)`` values.** Row order does not
    matter for COO construction, so counts can be streamed in whatever order it is stored in.
    A counts row whose key is not in ``match_index`` is dropped.

    The full (cells x genes) matrix is never materialized as a dense array.
    """
    n_cells = len(match_index)
    n_genes = len(gene_columns)
    seen = np.zeros(n_cells, dtype=bool)

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []

    # We will chunk the counts data into smaller pieces accorinding to a maximum bytes size to avoid loading
    # the entire dataset into memory at once. The +4 covers the two int64 identifier columns.
    chunk_rows = max(_COUNTS_CHUNK_BYTES // ((n_genes + 4) * np.dtype(_COUNTS_VALUE_DTYPE).itemsize), 1)
    log.debug("Streaming CosMx counts in chunks of %d row(s) for %d gene column(s).", chunk_rows, n_genes)

    # Only read the fov, cell_ID, and (already gene-filtered) gene columns of the expression
    # matrix, streaming it in chunks to avoid a dense (cells x genes) read.
    usecols = [counts_fov, counts_cell, *gene_columns]
    # Parsing the counts straight into the accumulation dtype halves each chunk versus pandas' inferred int64.
    gene_dtypes = dict.fromkeys(gene_columns, _COUNTS_VALUE_DTYPE)
    # The identifier columns must be declared too. read_csv infers dtypes independently per chunk,
    # so a single blank or non-numeric ID anywhere turns that chunk's identifiers into float64,
    # which then match nothing and silently drop every cell in the chunk. Declaring them turns a
    # malformed identifier into a hard error instead, while still accepting '1.0' as 1.
    id_dtypes = {counts_fov: np.int64, counts_cell: np.int64}
    with pd.read_csv(
        counts_path, header=0, usecols=usecols, dtype={**id_dtypes, **gene_dtypes}, chunksize=chunk_rows
    ) as reader:
        for chunk in reader:
            # Rebuild the same paired integer label ID used for match_index, since the vendor's
            # per-FOV cell ID alone is not unique across the dataset.
            chunk_keys = _vendor_label_ids(chunk[counts_fov].to_numpy(), chunk[counts_cell].to_numpy())
            # get_indexer resolves each key to its row position in one vectorized hash lookup, and
            # returns -1 for a counts row whose cell the metadata never mentions.
            row_index = match_index.get_indexer(chunk_keys).astype(_COUNTS_INDEX_DTYPE)
            matched = row_index >= 0
            if not matched.any():
                continue
            # A matched cell counts as seen even if all of its counts are zero.
            seen[row_index[matched]] = True

            # gene_columns is in file order, so this is a contiguous column subset.
            # Rows are left unfiltered; dropping unmatched COO triples below avoids a second dense copy.
            chunk_values = chunk[gene_columns].to_numpy()
            # That selection already copied the gene block, so release the parsed frame before the
            # nonzero scan rather than holding a second full-size copy of the chunk alongside it.
            del chunk

            # A sparse matrix only needs to record nonzero entries.
            # np.nonzero returns two 1D arrays of equal length,
            # ensuring that all nonzero entries have two coupled indices (row, col)
            nonzero_rows, nonzero_cols = np.nonzero(chunk_values)

            # Only keep the nonzero entries that correspond to a cell that was actually seen in the
            # metadata. Filtering before mapping through row_index avoids building a full-size
            # array of final row positions, most of which would be discarded again here.
            kept_entries = matched[nonzero_rows]
            if not kept_entries.any():
                continue
            kept_rows = nonzero_rows[kept_entries]
            kept_cols = nonzero_cols[kept_entries]

            # Select the row positions, column positions and values for those nonzero entries.
            row_chunks.append(row_index[kept_rows])
            col_chunks.append(kept_cols.astype(_COUNTS_INDEX_DTYPE))
            value_chunks.append(chunk_values[kept_rows, kept_cols])

    # Concatenate the accumulated COO triples into single arrays for the final sparse matrix construction.
    if value_chunks:
        # Release each list as soon as it has been concatenated: holding all three alongside their
        # full-size copies would double the accumulated COO footprint at exactly the peak.
        rows = np.concatenate(row_chunks)
        row_chunks.clear()
        cols = np.concatenate(col_chunks)
        col_chunks.clear()
        values = np.concatenate(value_chunks)
        value_chunks.clear()
    else:
        rows = np.array([], dtype=_COUNTS_INDEX_DTYPE)
        cols = np.array([], dtype=_COUNTS_INDEX_DTYPE)
        values = np.array([], dtype=_COUNTS_VALUE_DTYPE)
        log.warning(
            "CosMx counts file '%s' contains no nonzero entries after gene filtering, or contains "
            "no cell IDs that match the metadata. Check that the counts file and metadata file are "
            "compatible. The resulting sparse expression matrix will be empty with shape (%d, %d).",
            counts_path,
            n_cells,
            n_genes,
        )

    # Create a sparse matrix in Compressed Sparse Row format directly from the COO triples.
    X = csr_matrix((values, (rows, cols)), shape=(n_cells, n_genes))
    # A metadata cell that is absent in the counts file doesn't get excluded from the matrix,
    # because of shape=(n_cells, n_genes) with n_cells = len(match_index).
    # That is why the matrix is sliced down to the rows actually seen in counts, but only when some
    # cell was in fact missing
    return (X if seen.all() else X[seen]), seen


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
        raster_levels = [level_array.astype(np.uint32) for level_array in _read_cosmx_zarr_levels(path, kind="labels")]
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
    obtains the ordered level paths from that group's own OME-Zarr metadata, and exposes those
    levels as lazy Dask arrays using direct Zarr access.

    The group is opened once with ``zarr.open_group``, and is the source of both the metadata and
    the arrays.
    """
    zarr_group = zarr.open_group(path, mode="r")

    # OME-Zarr records its pyramid under a top-level "multiscales" attribute.
    multiscales = zarr_group.attrs.get("multiscales")
    if not multiscales:
        raise ValueError(f"CosMx {kind} Zarr store '{path}' does not contain a readable multiscale OME-Zarr dataset.")

    # Distinguish missing multiscale metadata from an ambiguous store describing several pyramids.
    if len(multiscales) > 1:
        raise ValueError(
            f"CosMx {kind} Zarr store '{path}' contains {len(multiscales)} multiscale nodes; expected exactly one."
        )

    # "datasets" is the ordered pyramid descriptor, level zero first; the vendor order is used as-is.
    datasets = multiscales[0].get("datasets")
    if not datasets:
        raise ValueError(f"CosMx {kind} Zarr store '{path}' has no readable multiscale metadata.")

    # Read each dataset lazily into a Dask array without changing the native pyramid levels.
    raster_levels: list[Array] = []
    for level_index, dataset in enumerate(datasets):
        dataset_path = dataset.get("path")
        # An entry without a "path" names no array at all, so the descriptor itself is malformed.
        if dataset_path is None:
            raise ValueError(
                f"CosMx {kind} Zarr store '{path}' has a multiscale dataset entry at position "
                f"{level_index} with no 'path' key, so the pyramid level it describes is unknown."
            )
        # A level named in the metadata but absent from the store is a corrupt export, not a KeyError.
        if dataset_path not in zarr_group:
            raise ValueError(
                f"CosMx {kind} Zarr store '{path}' lists pyramid level '{dataset_path}' in its multiscale "
                "metadata, but that array is not present in the store."
            )
        raster_levels.append(da.from_zarr(zarr_group[dataset_path]))

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

        # Convert each Dask array into a SpatialData-compatible xarray.DataArray
        parsed = model.parse(
            array,
            dims=dims,
            c_coords=channel_names,
        )
        # Wrap the DataArray in an xarray.Dataset, compatible with SpatialData's expectations for DataTree nodes.
        # Here, name="image" mirrors spatialdata's own internal convention
        # It has nothing to do with whether the element ends up registered as an image or a labels layer.
        levels[f"scale{level_index}"] = parsed.to_dataset(name="image")

    # Build the multiscale DataTree from the Datasets
    tree = DataTree.from_dict(levels)
    # The level-zero CosMx mosaic is already in the requested global pixel coordinate system,
    # so it receives an Identity() transform.
    # For the lower resolution levels, SpatialData derives scale factors from the shapes.
    _set_transformations(tree, {coordinate_system: Identity()})

    # Compute every level's pixel centres in the level-zero CosMx pixel coordinate system, needed xarray coordinate computations
    return compute_coordinates(tree)


def _load_fov_positions(fov_positions_path: Path | None) -> _FovPositions | None:
    """Read FOV tile origins and the mosaic top edge from ``fov_positions_file.csv``."""
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
            "transcript coordinates cannot be placed in the mosaic raster space.",
            fov_positions_path,
        )
        return None

    # Normalize the FOV keys before checking for duplicate vendor rows.
    fov_ids = _normalize_integer_ids(positions[fov_column])
    duplicate_fovs = fov_ids[fov_ids.duplicated()].unique()
    if duplicate_fovs.size:
        raise ValueError(
            f"CosMx FOV positions file '{fov_positions_path}' contains duplicate FOV(s) "
            f"{', '.join(str(fov) for fov in sorted(duplicate_fovs))}."
        )

    # Non-numeric values become NaN, so the finiteness check below catches them too.
    origins_x = pd.to_numeric(positions[x_column], errors="coerce").astype(float)
    origins_y = pd.to_numeric(positions[y_column], errors="coerce").astype(float)

    # Reject invalid origins before they become translations and corrupt allocation coordinates.
    invalid = ~(np.isfinite(origins_x) & np.isfinite(origins_y))
    if invalid.any():
        raise ValueError(
            f"CosMx FOV positions file '{fov_positions_path}' contains a non-numeric or non-finite "
            f"origin for FOV(s) {', '.join(str(fov) for fov in sorted(fov_ids[invalid].unique()))}."
        )

    # The vendor records each tile's *top* edge in a y-up system, so the mosaic's first raster row
    # sits at the largest origin y. That constant is what converts global pixel y into raster rows.
    return _FovPositions(
        origins=dict(zip(fov_ids.tolist(), zip(origins_x.tolist(), origins_y.tolist(), strict=True), strict=True)),
        top_global_y=float(origins_y.max()),
    )


def _apply_global_coordinates(
    partition: pd.DataFrame,
    gene_column: str,
    global_x: str,
    global_y: str,
    top_global_y: float | None,
    transcripts_path: Path,
) -> pd.DataFrame:
    """Place global pixel transcript coordinates in mosaic raster pixels, fully vectorized."""
    # Preserve untouched vendor columns while replacing source columns with Sparrow's canonical names.
    placed = partition.rename(columns={gene_column: _GENES_KEY, global_x: "x", global_y: "y"})
    # The global pixel x axis already runs the same way as raster columns.
    placed["x"] = partition[global_x].astype(float)
    if top_global_y is None:
        # The mosaic top edge is unknown, so the vendor's upward y is kept as-is; the caller has
        # already warned that the points will not overlay the rasters.
        placed["y"] = partition[global_y].astype(float)
    else:
        # Vendor global y increases upward while raster rows increase downward.
        placed["y"] = top_global_y - partition[global_y].astype(float)

    _reject_non_finite_coordinates(placed, transcripts_path)

    return placed


def _apply_fov_origins(
    partition: pd.DataFrame,
    gene_column: str,
    local_x: str,
    local_y: str,
    fov_column: str,
    origin_x: pd.Series,
    origin_row: pd.Series,
    transcripts_path: Path,
) -> pd.DataFrame:
    """Translate local transcript coordinates into mosaic raster pixels, fully vectorized."""
    # Map off integers, not strings to be more efficient
    # Stringifying the FOV column would allocate a Python string per transcript
    fov_ids = _normalize_integer_ids(partition[fov_column])
    partition_origin_x = fov_ids.map(origin_x)
    partition_origin_row = fov_ids.map(origin_row)

    # Validate that every FOV in this partition has a corresponding FOV origin.
    missing = partition_origin_x.isna() | partition_origin_row.isna()
    if missing.any():
        missing_origin_fovs = sorted(fov_ids[missing].unique())
        raise ValueError(
            "CosMx FOV positions metadata is incomplete; missing origins for FOVs "
            f"{', '.join(str(fov) for fov in missing_origin_fovs)}. "
            "Valid origins are required for every local transcript FOV."
        )

    # Preserve untouched vendor columns while replacing source columns with Sparrow's canonical names.
    translated = partition.rename(columns={gene_column: _GENES_KEY, local_x: "x", local_y: "y"})
    # Both axes are plain offsets here: local pixels already run left-to-right and top-to-bottom
    # within their tile, and origin_row is that tile's first row in the mosaic.
    translated["x"] = partition[local_x].astype(float) + partition_origin_x
    translated["y"] = partition[local_y].astype(float) + partition_origin_row

    _reject_non_finite_coordinates(translated, transcripts_path)

    return translated


def _reject_non_finite_coordinates(partition: pd.DataFrame, transcripts_path: Path) -> None:
    """Raise when a placed partition holds a NaN or infinite transcript coordinate."""
    # Checking inside the partition function keeps this in the Dask graph
    non_finite = ~(np.isfinite(partition["x"].to_numpy()) & np.isfinite(partition["y"].to_numpy()))
    if non_finite.any():
        raise ValueError(
            f"CosMx transcript file '{transcripts_path}' contains {int(non_finite.sum())} "
            "transcript(s) with a non-finite x or y coordinate in a single partition. "
            "Every transcript must have a finite position."
        )


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


def _vendor_label_ids(fov_ids: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    """Reconstruct the CosMx ``CellLabels`` mask value for each ``(fov, cell_ID)`` pair.

    The vendor numbers its global label mosaic with Szudzik's elegant pairing function, an
    injective map from a pair of non-negative integers onto a single one:

        pair(x, y) = y*y + x            if x <  y
        pair(x, y) = x*x + x + y        if x >= y

    applied as ``pair(fov, cell_ID)``. Because the pairing is injective, the resulting key is
    unique across the dataset even though the vendor's ``cell_ID`` restarts at 1 in every FOV.
    The key is also the integer actually stored in the label mask, so the table can be joined to
    its labels layer without ever reading the mask.
    """
    # Work in int64: cell_ID squared comfortably exceeds the uint32 label range during computation.
    fovs = np.asarray(fov_ids, dtype=np.int64)
    cells = np.asarray(cell_ids, dtype=np.int64)
    return np.where(fovs < cells, cells * cells + fovs, fovs * fovs + fovs + cells)


def _normalize_integer_ids(values: pd.Series) -> pd.Series:
    """Normalise a vendor identifier column to plain integers, so ``1``, ``1.0`` and ``001`` all become ``1``."""
    if pd.api.types.is_integer_dtype(values):
        return values.astype("int64")

    # Text and float encodings round-trip through a numeric dtype before being truncated to int.
    return pd.to_numeric(values, errors="raise").astype("int64")
