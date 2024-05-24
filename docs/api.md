# API

```{eval-rst}

Import SPArrOW as::

    import sparrow as sp

.. module:: sparrow
```

## IO

I/O.

```{eval-rst}

.. module:: sparrow.io
.. currentmodule:: sparrow

.. autosummary::
    :toctree: generated

    io.create_sdata
    io.read_transcripts
    io.read_resolve_transcripts
    io.read_vizgen_transcripts
    io.read_stereoseq_transcripts

```

## Image

Operations on image and labels layers.

```{eval-rst}

.. module:: sparrow.im
.. currentmodule:: sparrow

.. autosummary::
    :toctree: generated

    im.apply
    im.tiling_correction
    im.enhance_contrast
    im.min_max_filtering
    im.transcript_density
    im.combine
    im.segment
    im.segment_points
    im.expand_labels_layer
    im.align_labels_layers
    im.apply_labels_layers
    im.filter_labels_layer
    im.mask_to_original
    im.merge_labels_layers
    im.merge_labels_layers_nuclei
```

## Shape

Operations on shapes (polygons) layers.

```{eval-rst}

.. module:: sparrow.sh
.. currentmodule:: sparrow

.. autosummary::
    :toctree: generated

    sh.create_voronoi_boundaries
```

## Table

Operations on table (`AnnData` object) layers.

```{eval-rst}

.. module:: sparrow.tb
.. currentmodule:: sparrow

.. autosummary::
    :toctree: generated

    tb.allocate
    tb.preprocess_anndata
    tb.filter_on_size
    tb.cluster
    tb.score_genes
    tb.correct_marker_genes
    tb.cluster_cleanliness
    tb.nhood_enrichment
    tb.allocate_intensity
    tb.add_regionprop_features
```

## Plotting

Plotting functions.

```{eval-rst}

.. module:: sparrow.pl
.. currentmodule:: sparrow

.. autosummary::
    :toctree: generated

    pl.plot_image
    pl.plot_shapes
    pl.plot_labels
    pl.tiling_correction
    pl.flatfield
    pl.segment
    pl.sanity_plot_transcripts_matrix
    pl.analyse_genes_left_out
    pl.transcript_density
    pl.preprocess_anndata
    pl.cluster
    pl.score_genes
    pl.cluster_cleanliness
    pl.nhood_enrichment
```
