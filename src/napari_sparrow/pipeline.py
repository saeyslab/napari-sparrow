""" This file contains the six pipeline steps that are used by the single pipeline.
Some steps consist of multiple substeps"""

import os
from typing import Dict, List, Tuple

from omegaconf import DictConfig, ListConfig
from spatialdata import SpatialData

import napari_sparrow as nas

log = nas.utils.get_pylogger(__name__)


def load(cfg: DictConfig) -> SpatialData:
    """Loading step, the first step of the pipeline, performs creation of spatial data object."""

    # cast to list if cfg.dataset.image is a ListConfig object (i.e. for multiple channels)
    if isinstance(cfg.dataset.image, ListConfig):
        filename_pattern = list(cfg.dataset.image)
    else:
        filename_pattern = cfg.dataset.image

    log.info( "Creating sdata." )
    sdata = nas.io.create_sdata(
        input=filename_pattern,
        output_path=os.path.join(cfg.paths.output_dir, "sdata.zarr"),
        img_layer="raw_image",
        crd=None,
        chunks=1024,  # TODO make chunks configurable
    )
    log.info( "Finished creating sdata." )

    return sdata


def clean(cfg: DictConfig, sdata: SpatialData) -> SpatialData:
    """Cleaning step, the second step of the pipeline, performs tilingCorrection and preprocessing of the image to improve image quality."""

    nas.pl.plot_image(
        sdata=sdata,
        output=os.path.join(cfg.paths.output_dir, "original"),
        crd=cfg.clean.small_size_vis,
        img_layer="raw_image",
    )

    # Perform tilingCorrection on the whole image, corrects illumination and performs inpainting
    if cfg.clean.tilingCorrection:
        log.info("Start tiling correction.")

        sdata, flatfields = nas.im.tiling_correction(
            sdata=sdata,
            crd=cfg.clean.crop_param if cfg.clean.crop_param is not None else None,
            tile_size=cfg.clean.tile_size,
            output_layer="tiling_correction",
        )

        log.info("Tiling correction finished.")

        # Write plot to given path if output is enabled
        if "tiling_correction" in cfg.paths:
            log.info(f"Writing tiling correction plot to {cfg.paths.tiling_correction}")
            nas.pl.tiling_correction(
                sdata=sdata,
                img_layer=["raw_image", "tiling_correction"],
                crd=cfg.clean.small_size_vis
                if cfg.clean.small_size_vis is not None
                else None,
                output=cfg.paths.tiling_correction,
            )
            for i, flatfield in enumerate(flatfields):
                # flatfield can be None is tiling correction failed.
                if flatfield is not None:
                    nas.pl.flatfield(
                        flatfield, output=f"{cfg.paths.tiling_correction}_flatfield_{i}"
                    )

        nas.pl.plot_image(
            sdata=sdata,
            output=os.path.join(cfg.paths.output_dir, "tiling_correction"),
            crd=cfg.clean.small_size_vis,
            img_layer="tiling_correction",
        )

    # min max filtering

    if cfg.clean.minmaxFiltering:
        log.info("Start min max filtering.")

        sdata = nas.im.min_max_filtering(
            sdata=sdata,
            size_min_max_filter=list(cfg.clean.size_min_max_filter)
            if isinstance(cfg.clean.size_min_max_filter, ListConfig)
            else cfg.clean.size_min_max_filter,
        )

        log.info("Min max filtering finished.")

        nas.pl.plot_image(
            sdata=sdata,
            output=os.path.join(cfg.paths.output_dir, "min_max_filtered"),
            crd=cfg.clean.small_size_vis,
            img_layer="min_max_filtered",
        )

    # contrast enhancement

    if cfg.clean.contrastEnhancing:
        log.info("Start contrast enhancing.")

        sdata = nas.im.enhance_contrast(
            sdata=sdata,
            contrast_clip=list(cfg.clean.contrast_clip)
            if isinstance(cfg.clean.contrast_clip, ListConfig)
            else cfg.clean.contrast_clip,
            chunks=cfg.clean.chunksize_clahe,
            depth=cfg.clean.depth,
        )

        log.info("Contrast enhancing finished.")

        nas.pl.plot_image(
            sdata=sdata,
            output=os.path.join(cfg.paths.output_dir, "clahe"),
            crd=cfg.clean.small_size_vis,
            img_layer="clahe",
        )

    return sdata


def segment(cfg: DictConfig, sdata: SpatialData) -> SpatialData:
    """Segmentation step, the third step of the pipeline, performs cellpose segmentation and creates masks."""

    log.info("Start segmentation.")

    # Perform segmentation
    sdata = nas.im.segmentation_cellpose(
        sdata=sdata,
        output_layer=cfg.segmentation.output_layer,
        crd=cfg.segmentation.crop_param,
        device=cfg.device,
        min_size=cfg.segmentation.min_size,
        flow_threshold=cfg.segmentation.flow_threshold,
        diameter=cfg.segmentation.diameter,
        cellprob_threshold=cfg.segmentation.cellprob_threshold,
        model_type=cfg.segmentation.model_type,
        channels=cfg.segmentation.channels,
        chunks=cfg.segmentation.chunks,
        lazy=cfg.segmentation.lazy,
    )

    log.info("Segmentation finished.")

    shapes_layer = None
    for key in sdata.shapes.keys():
        if "boundaries" in key:
            shapes_layer = key
            break

    # plot the image in SpatialData object in last position, will typically be 'clahe'
    img_layer = [*sdata.images][-1]

    nas.pl.segment(
        sdata=sdata,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        output=cfg.paths.segmentation,
    )

    if cfg.segmentation.voronoi_radius:
        sdata = nas.sh.create_voronoi_boundaries(
            sdata,
            radius=cfg.segmentation.voronoi_radius,
            shapes_layer=shapes_layer,
        )
        nas.pl.segment(
            sdata=sdata,
            crd=cfg.segmentation.small_size_vis
            if cfg.segmentation.small_size_vis is not None
            else cfg.clean.small_size_vis,
            img_layer=img_layer,
            shapes_layer="expanded_cells" + str(cfg.segmentation.voronoi_radius),
            output=f"{cfg.paths.segmentation}_expanded_cells_{cfg.segmentation.voronoi_radius}",
        )

    return sdata


def allocate(cfg: DictConfig, sdata: SpatialData) -> SpatialData:
    """Allocation step, the fourth step of the pipeline, creates the adata object from the mask and allocates the transcripts from the supplied file."""

    sdata = nas.io.read_transcripts(
        sdata,
        path_count_matrix=cfg.dataset.coords,
        path_transform_matrix=cfg.dataset.transform_matrix,
        delimiter=cfg.allocate.delimiter,
        header=cfg.allocate.header,
        column_x=cfg.allocate.column_x,
        column_y=cfg.allocate.column_y,
        column_gene=cfg.allocate.column_gene,
        column_midcount=cfg.allocate.column_midcount,
        debug=cfg.allocate.debug,
    )

    if cfg.segmentation.voronoi_radius:
        shapes_layer = "expanded_cells" + str(cfg.segmentation.voronoi_radius)
    else:
        for key in sdata.shapes.keys():
            if "boundaries" in key:
                shapes_layer = key
                break

    log.info("Start allocation.")

    sdata = nas.tb.allocate(
        sdata=sdata,
        shapes_layer=shapes_layer,
    )

    log.info("Allocation finished.")

    # plot the image in SpatialData object in last position, will typically be 'clahe'
    img_layer = [*sdata.images][-1]

    nas.pl.plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        output=cfg.paths.polygons,
    )

    nas.pl.analyse_genes_left_out(
        sdata,
        labels_layer=cfg.segmentation.output_layer,
        output=cfg.paths.analyse_genes_left_out,
    )

    log.info("Preprocess AnnData.")

    # Perform normalization based on size + all cells with less than 10 genes and all genes with less than 5 cells are removed.
    sdata = nas.tb.preprocess_anndata(
        sdata,
        min_counts=cfg.allocate.min_counts,
        min_cells=cfg.allocate.min_cells,
        size_norm=cfg.allocate.size_norm,
        n_comps=cfg.allocate.n_comps,
        shapes_layer=shapes_layer,
    )

    log.info("Preprocessing AnnData finished.")

    nas.pl.preprocess_anndata(
        sdata,
        output=cfg.paths.preprocess_adata,
    )

    nas.pl.plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        column=cfg.allocate.total_counts_column,
        cmap=cfg.allocate.total_counts_cmap,
        alpha=cfg.allocate.total_counts_alpha,
        output=cfg.paths.total_counts,
    )

    # Filter all cells based on size and distance
    sdata = nas.tb.filter_on_size(
        sdata,
        min_size=cfg.allocate.min_size,
        max_size=cfg.allocate.max_size,
    )

    nas.pl.plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        column=cfg.allocate.shape_size_column,
        cmap=cfg.allocate.shape_size_cmap,
        alpha=cfg.allocate.shape_size_alpha,
        output=cfg.paths.shape_size,
    )

    log.info("Start clustering")

    sdata = nas.tb.cluster(
        sdata,
        pcs=cfg.allocate.pcs,
        neighbors=cfg.allocate.neighbors,
        cluster_resolution=cfg.allocate.cluster_resolution,
    )

    log.info("Clustering finished")

    nas.pl.cluster(
        sdata,
        output=cfg.paths.cluster,
    )

    nas.pl.plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        column=cfg.allocate.leiden_column,
        cmap=cfg.allocate.leiden_cmap,
        alpha=cfg.allocate.leiden_alpha,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        output=cfg.paths.leiden,
    )

    return sdata


def annotate(
    cfg: DictConfig, sdata: SpatialData
) -> Tuple[SpatialData, Dict[str, List[str]]]:
    """Annotation step, the fifth step of the pipeline, annotates the cells with celltypes based on the marker genes file."""

    # Get arguments from cfg else empty objects
    repl_columns = (
        cfg.annotate.repl_columns if "repl_columns" in cfg.annotate else dict()
    )
    del_celltypes = (
        cfg.annotate.del_celltypes if "del_celltypes" in cfg.annotate else []
    )

    # Load marker genes, replace columns with different name, delete genes from list

    log.info( "Start scoring genes" )

    mg_dict, scoresper_cluster = nas.tb.score_genes(
        sdata=sdata,
        path_marker_genes=cfg.dataset.markers,
        delimiter=cfg.annotate.delimiter,
        row_norm=cfg.annotate.row_norm,
        repl_columns=repl_columns,
        del_celltypes=del_celltypes,
    )

    log.info( "Scoring genes finished" )

    if cfg.segmentation.voronoi_radius:
        shapes_layer = "expanded_cells" + str(cfg.segmentation.voronoi_radius)
    else:
        for key in sdata.shapes.keys():
            if "boundaries" in key:
                shapes_layer = key
                break

    # plot the image in SpatialData object in last position, will typically be 'clahe'
    img_layer = [*sdata.images][-1]

    nas.pl.score_genes(
        sdata=sdata,
        scoresper_cluster=scoresper_cluster,
        shapes_layer=shapes_layer,
        img_layer=img_layer,
        output=cfg.paths.score_genes,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
    )

    return sdata, mg_dict


def visualize(
    cfg: DictConfig, sdata: SpatialData, mg_dict: Dict[str, List[str]]
) -> SpatialData:
    """Visualisation step, the sixth and final step of the pipeline, checks the cluster cleanliness and performs nhood enrichement before saving the data as SpatialData object."""

    # Perform correction for transcripts (and corresponding celltypes) that occur in all cells and are overexpressed
    if "correct_marker_genes_dict" in cfg.visualize:
        sdata = nas.tb.correct_marker_genes(
            sdata,
            celltype_correction_dict=cfg.visualize.correct_marker_genes_dict,
        )

    # Get arguments from cfg else None objects
    celltype_indexes = (
        cfg.visualize.celltype_indexes if "celltype_indexes" in cfg.visualize else None
    )
    colors = cfg.visualize.colors if "colors" in cfg.visualize else None

    # Check cluster cleanliness
    sdata, color_dict = nas.tb.cluster_cleanliness(
        sdata,
        celltypes=list(mg_dict.keys()),
        celltype_indexes=celltype_indexes,
        colors=colors,
    )

    if cfg.segmentation.voronoi_radius:
        shapes_layer = "expanded_cells" + str(cfg.segmentation.voronoi_radius)
    else:
        for key in sdata.shapes.keys():
            if "boundaries" in key:
                shapes_layer = key
                break

    # plot the image in SpatialData object in last position, will typically be 'clahe'
    img_layer = [*sdata.images][-1]

    nas.pl.cluster_cleanliness(
        sdata=sdata,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        color_dict=color_dict,
        output=cfg.paths.cluster_cleanliness,
    )

    if cfg.visualize.calculate_transcript_density:
        # calculate transcript density
        sdata = nas.im.transcript_density(
            sdata,
            img_layer=img_layer,
            crd=cfg.segmentation.crop_param,
            output_layer="transcript_density",
        )

        nas.pl.transcript_density(
            sdata,
            img_layer=[img_layer, "transcript_density"],
            crd=cfg.segmentation.small_size_vis
            if cfg.segmentation.small_size_vis is not None
            else cfg.clean.small_size_vis,
            output=cfg.paths.transcript_density,
        )

    # calculate nhood enrichment
    sdata = nas.tb.nhood_enrichment(sdata)
    nas.pl.nhood_enrichment(
        sdata,
        output=cfg.paths.nhood,
    )

    return sdata
