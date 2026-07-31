from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from spatialdata import SpatialData

from sparrow.table._table import ProcessTable, add_table_layer
from sparrow.utils._keys import _CELLSIZE_KEY, _TRANSCRIPT_DENSITY_KEY
from sparrow.utils.pylogger import get_pylogger

# Set up a module-scoped logger for warnings about count-source assumptions.
log = get_pylogger(__name__)


def add_transcript_density(
    sdata: SpatialData,
    labels_layer: str | Iterable[str],
    table_layer: str,
    pixel_size: float,
    output_layer: str | None = None,
    counts_layer: str | None = None,
    cellsize_key: str = _CELLSIZE_KEY,
    density_key: str = _TRANSCRIPT_DENSITY_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """Add per-cell transcript density to an AnnData table.

    Transcript density is calculated as the number of transcripts assigned to
    each cell divided by its physical mask area in square micrometers.

    Parameters
    ----------
    sdata
        SpatialData object containing the labels and AnnData table.
    labels_layer
        Labels layer or layers whose linked observations should be processed.
        ``labels_layer`` lets Sparrow validate uniqueness per region instead of requiring globally unique IDs.
        The current requirement follows Sparrow’s existing table-processing conventions and prevents accidentally calculating density across unrelated masks or samples.
    table_layer
        AnnData table containing one observation per cell or mask.
    pixel_size
        Physical size of one mask pixel in micrometers. Mask areas in
        ``cellsize_key`` are assumed to be expressed in pixels.
    output_layer
        Name of the output table layer. If ``None``, ``"<table_layer>_density"``
        is used. Set this equal to ``table_layer`` and set ``overwrite=True``
        to add the density column directly to the existing table layer. For a
        table containing multiple labels regions, use a separate output layer
        unless all regions are included in ``labels_layer``; otherwise the
        selected regions replace the existing table contents.
    counts_layer
        Optional AnnData layer containing raw transcript counts. If ``None``,
        the ``total_counts`` observation column is used when available;
        otherwise the rows of ``adata.X`` are summed and a warning is emitted.
        The fallback is only a true transcript count when ``adata.X`` contains
        unnormalized counts.
    cellsize_key
        Observation column containing mask areas in pixels.
    density_key
        Observation column in which to store transcript density.
    overwrite
        If True, overwrite an existing output table layer.

    Returns
    -------
    The updated SpatialData object with transcript density added to the output
    AnnData table layer.

    Raises
    ------
    ValueError
        If ``pixel_size`` is not positive and finite, if the mask-area column
        or requested count layer is missing, or if any mask area or count is
        invalid.

    Examples
    --------
    >>> sdata = sparrow.tb.add_transcript_density(
    ...     sdata,
    ...     labels_layer="segmentation_mask",
    ...     table_layer="table_transcriptomics_preprocessed",
    ...     pixel_size=0.138,
    ...     counts_layer="raw_counts",
    ...     output_layer="table_transcriptomics_density",
    ...     overwrite=True,
    ... )
    >>> # For a single-region table, add the column directly to the existing layer.
    >>> sdata = sparrow.tb.add_transcript_density(
    ...     sdata,
    ...     labels_layer="segmentation_mask",
    ...     table_layer="table_transcriptomics_preprocessed",
    ...     output_layer="table_transcriptomics_preprocessed",
    ...     pixel_size=0.138,
    ...     counts_layer="raw_counts",
    ...     overwrite=True,
    ... )
    """
    # Validate the pixel resolution before reading or copying the table.
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("'pixel_size' must be a positive, finite number in micrometers per pixel.")

    # Select and copy only the observations linked to the requested labels.
    # Because labels_layer was supplied, it filters the table to only observations whose fov_labels value matches the requested labels.
    # It also preserves the SpatialData table-region metadata.
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()

    # Ensure that the table contains the pixel-area feature required for conversion to physical area.
    if cellsize_key not in adata.obs:
    # Because labels_layer was supplied, it filters the table to only observations whose fov_labels value matches the requested labels.
    # It also preserves the SpatialData table-region metadata.e_key not in adata.obs:
        raise ValueError(
            f"Observation column '{cellsize_key}' is missing from table layer '{table_layer}'. "
            "Run preprocessing or provide a table containing mask areas before calculating transcript density."
        )

    # Convert pixel areas to numeric values so invalid or missing measurements can be rejected explicitly.
    mask_sizes = pd.to_numeric(adata.obs[cellsize_key], errors="coerce").to_numpy(dtype=float)

    # Reject empty, missing, infinite, or negative mask areas before division.
    if np.any(~np.isfinite(mask_sizes)) or np.any(mask_sizes <= 0):
        raise ValueError(f"Observation column '{cellsize_key}' must contain positive, finite mask areas in pixels.")

    # Read the requested raw-count matrix when the caller supplied an AnnData layer name.
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise ValueError(f"Counts layer '{counts_layer}' is not present in table layer '{table_layer}'.")
        # Assuming counts_layer="raw_counts" is a matrix of raw transcript counts of shape cells x genes
        # Summing across genes to get total transcript counts per cell.
        transcript_counts = np.asarray(adata.layers[counts_layer].sum(axis=1)).ravel().astype(float)

    # Prefer Scanpy's raw total-count metric when it is present on a preprocessed table.
    elif "total_counts" in adata.obs:
        transcript_counts = pd.to_numeric(adata.obs["total_counts"], errors="coerce").to_numpy(dtype=float)

    # Warn that summing .X is only valid when .X still contains raw counts.
    else:
        log.warning(
            "No 'counts_layer' was provided and 'total_counts' is unavailable. "
            "Falling back to summing 'adata.X' for transcript counts. This is "
            "valid only when 'adata.X' contains unnormalized counts; if it "
            "contains normalized, log-transformed, or scaled values, the "
            "resulting transcript density will not represent transcripts per "
            "square micrometer. Pass counts_layer='raw_counts' when available."
        )

        # Sum the table matrix across genes for an unprocessed allocation table.
        transcript_counts = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)

    # Reject invalid transcript totals before calculating densities.
    if np.any(~np.isfinite(transcript_counts)) or np.any(transcript_counts < 0):
        raise ValueError("Transcript counts must be finite and non-negative.")

    # Convert pixel mask areas to square micrometers using the platform resolution.
    mask_areas_um2 = mask_sizes * pixel_size**2

    # Store transcript counts per physical mask area as an observation-level feature.
    adata.obs[density_key] = transcript_counts / mask_areas_um2

    # Derive a table name that keeps the source table unchanged when no output name is supplied.
    if output_layer is None:
        output_layer = f"{table_layer}_density"

    # Persist the annotated table using the same table-layer metadata conventions as other table operations.
    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    # Return the same SpatialData object so the operation can be chained with other Sparrow functions.
    return sdata