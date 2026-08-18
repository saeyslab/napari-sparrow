from __future__ import annotations

import numpy as np
import pytest
from spatialdata import SpatialData

from sparrow.table._preprocess import preprocess_transcriptomics
from sparrow.table._transcript_density import add_transcript_density
from sparrow.utils._keys import _CELLSIZE_KEY, _TRANSCRIPT_DENSITY_KEY


def test_add_transcript_density_uses_raw_counts(sdata_transcripts_no_backed: SpatialData) -> None:
    # Create a table containing mask areas and a raw-count layer while normalizing its main matrix.
    sdata_transcripts_no_backed = preprocess_transcriptomics(
        sdata_transcripts_no_backed,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics",
        output_layer="table_transcriptomics_density_input",
        min_counts=0,
        min_cells=0,
        size_norm=False,
        n_comps=10,
        overwrite=True,
    )

    # Add transcript density to a separate table layer using the unnormalized counts.
    sdata_transcripts_no_backed = add_transcript_density(
        sdata_transcripts_no_backed,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics_density_input",
        pixel_size=0.5,
        counts_layer="raw_counts",
        output_layer="table_transcriptomics_density",
        overwrite=True,
    )

    # Read the source and density tables for comparing the calculated values.
    source_adata = sdata_transcripts_no_backed.tables["table_transcriptomics_density_input"]
    density_adata = sdata_transcripts_no_backed.tables["table_transcriptomics_density"]

    # Calculate the expected density from raw transcript counts and square-micrometer areas.
    raw_counts = np.asarray(source_adata.layers["raw_counts"].sum(axis=1)).ravel()
    mask_areas_um2 = source_adata.obs[_CELLSIZE_KEY].to_numpy(dtype=float) * 0.5**2
    expected_density = raw_counts / mask_areas_um2

    # Verify values, observation alignment
    np.testing.assert_allclose(density_adata.obs[_TRANSCRIPT_DENSITY_KEY].to_numpy(), expected_density)
    assert np.array_equal(density_adata.obs_names, source_adata.obs_names)
    # Verify source-table immutability
    assert _TRANSCRIPT_DENSITY_KEY not in source_adata.obs


@pytest.mark.parametrize("pixel_size", [0.0, -0.1, float("nan")])
def test_add_transcript_density_rejects_invalid_pixel_size(
    sdata_transcripts_no_backed: SpatialData,
    pixel_size: float,
) -> None:
    # Reject invalid physical resolutions before attempting to access the table.
    with pytest.raises(ValueError, match="pixel_size"):
        add_transcript_density(
            sdata_transcripts_no_backed,
            labels_layer="segmentation_mask",
            table_layer="table_transcriptomics",
            pixel_size=pixel_size,
        )