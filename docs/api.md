# API

```{eval-rst}

Import napari_sparrow as::

    import napari_sparrow as nas

.. module:: napari_sparrow
```


## IO

I/O.

```{eval-rst}

.. module:: napari_sparrow.io
.. currentmodule:: napari_sparrow

.. autosummary::
    :toctree: generated

    io.create_sdata
    io.read_transcripts
    io.read_resolve_transcripts
    io.read_vizgen_transcripts
    io.read_stereoseq_transcripts

```

## Image

Operations on images.

```{eval-rst}

.. module:: napari_sparrow.im
.. currentmodule:: napari_sparrow

.. autosummary::
    :toctree: generated

    im.apply
    im.tiling_correction
    im.enhance_contrast
    im.min_max_filtering
    im.segment
    im.transcript_density
```

## Shape

Operations on shapes (polygons).

```{eval-rst}

.. module:: napari_sparrow.sh
.. currentmodule:: napari_sparrow

.. autosummary::
    :toctree: generated

    sh.create_voronoi_boundaries
```

## Table

Operations on tables (`AnnData` object).

```{eval-rst}

.. module:: napari_sparrow.tb
.. currentmodule:: napari_sparrow

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
```

## Plotting

Plotting functions.

```{eval-rst}

.. module:: napari_sparrow.pl
.. currentmodule:: napari_sparrow

.. autosummary::
    :toctree: generated

    pl.plot_image
    pl.plot_shapes
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