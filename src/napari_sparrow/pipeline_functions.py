""" This file contains the five pipeline steps that are used by the single pipeline.
Some steps consist of multiple substeps from the functions file. """

import os
from pathlib import Path
from typing import Dict, List, Tuple

import squidpy.im as sq
from omegaconf import DictConfig, ListConfig
from skimage import io
from spatialdata import SpatialData

from napari_sparrow import functions as fc
from napari_sparrow import utils

log = utils.get_pylogger(__name__)


def load(cfg: DictConfig) -> SpatialData:
    """Loading step, the first step of the pipeline, performs creation of spatial data object."""

    layer_name = "raw_image"

    # cast to list if cfg.dataset.image is a ListConfig object (i.e. for multiple channels)
    if isinstance( cfg.dataset.image, ListConfig ):
        filename_pattern=list(cfg.dataset.image)
    else:
        filename_pattern=cfg.dataset.image

    sdata = fc.create_sdata(
        filename_pattern=filename_pattern,
        output_path=os.path.join(cfg.paths.output_dir, "sdata.zarr"),
        layer_name=layer_name,
        chunks=1024,  # TODO make chunks configurable
    )
    return sdata


def clean(cfg: DictConfig, sdata: SpatialData) -> SpatialData:
    """Cleaning step, the second step of the pipeline, performs tilingCorrection and preprocessing of the image to improve image quality."""

    fc.plot_image_container(
        sdata=sdata,
        output_path=os.path.join(cfg.paths.output_dir, "original"),
        crd=cfg.clean.small_size_vis,
        layer="raw_image",
    )

    # Perform tilingCorrection on the whole image, corrects illumination and performs inpainting
    if cfg.clean.tilingCorrection:
        sdata, flatfields = fc.tilingCorrection(
            sdata=sdata,
            crop_param=cfg.clean.crop_param
            if cfg.clean.crop_param is not None
            else None,
            tile_size=cfg.clean.tile_size,
            output_layer="tiling_correction",
        )

        # Write plot to given path if output is enabled
        if "tiling_correction" in cfg.paths:
            log.info(f"Writing flatfield plot to {cfg.paths.tiling_correction}")
            for i,channel in enumerate(sdata[ "tiling_correction" ].c.data):

                fc.tilingCorrectionPlot(
                    img=sdata[ "tiling_correction" ].isel(c=channel).to_numpy(),
                    flatfield=flatfields[i],
                    img_orig=sdata[ "tiling_correction" ].isel(c=channel).to_numpy(),
                    output=f"{cfg.paths.tiling_correction}_{channel}_",
                )

        fc.plot_image_container(
            sdata=sdata,
            output_path=os.path.join(cfg.paths.output_dir, "tiling_correction"),
            crd=cfg.clean.small_size_vis,
            layer="tiling_correction",
        )

    # tophat filtering

    if cfg.clean.tophatFiltering:
        sdata = fc.tophat_filtering(
            sdata=sdata,
            size_tophat=cfg.clean.size_tophat,
        )

        fc.plot_image_container(
            sdata=sdata,
            output_path=os.path.join(cfg.paths.output_dir, "tophat_filtered"),
            crd=cfg.clean.small_size_vis,
            layer="tophat_filtered",
        )

    # clahe processing

    if cfg.clean.claheProcessing:
        sdata = fc.clahe_processing(
            sdata=sdata,
            contrast_clip=cfg.clean.contrast_clip,
            chunksize_clahe=cfg.clean.chunksize_clahe,
        )

        fc.plot_image_container(
            sdata=sdata,
            output_path=os.path.join(cfg.paths.output_dir, "clahe"),
            crd=cfg.clean.small_size_vis,
            layer="clahe",
        )

    return sdata


def segment(cfg: DictConfig, sdata: SpatialData) -> SpatialData:
    """Segmentation step, the third step of the pipeline, performs cellpose segmentation and creates masks."""

    # Perform segmentation
    sdata = fc.segmentation_cellpose(
        sdata=sdata,
        output_layer=cfg.segmentation.output_layer,
        crop_param=cfg.segmentation.crop_param,
        device=cfg.device,
        min_size=cfg.segmentation.min_size,
        flow_threshold=cfg.segmentation.flow_threshold,
        diameter=cfg.segmentation.diameter,
        cellprob_threshold=cfg.segmentation.cellprob_threshold,
        model_type=cfg.segmentation.model_type,
        channels=cfg.segmentation.channels,
        chunks=cfg.segmentation.chunks,
        lazy=True,
    )

    for key in sdata.shapes.keys():
        if "boundaries" in key:
            shapes_layer = key
            break

    fc.segmentationPlot(
        sdata=sdata,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        shapes_layer=shapes_layer,
        output=cfg.paths.segmentation,
    )

    if cfg.segmentation.voronoi_radius:
        sdata = fc.create_voronoi_boundaries(
            sdata,
            radius=cfg.segmentation.voronoi_radius,
            shapes_layer=shapes_layer,
        )
        fc.segmentationPlot(
            sdata=sdata,
            crd=cfg.segmentation.small_size_vis
            if cfg.segmentation.small_size_vis is not None
            else cfg.clean.small_size_vis,
            shapes_layer="expanded_cells" + str(cfg.segmentation.voronoi_radius),
            output=f"{cfg.paths.segmentation}_expanded_cells_{cfg.segmentation.voronoi_radius}",
        )

    return sdata


def allocate(cfg: DictConfig, sdata: SpatialData) -> SpatialData:
    """Allocation step, the fourth step of the pipeline, creates the adata object from the mask and allocates the transcripts from the supplied file."""

    sdata = fc.read_transcripts(
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

    sdata = fc.allocation(
        sdata=sdata,
        shapes_layer=shapes_layer,
    )

    fc.plot_shapes(
        sdata,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        output=cfg.paths.polygons,
    )

    # Perform normalization based on size + all cells with less than 10 genes and all genes with less than 5 cells are removed.
    sdata = fc.preprocessAdata(
        sdata,
        nuc_size_norm=cfg.allocate.nuc_size_norm,
        shapes_layer=shapes_layer,
    )

    fc.preprocesAdataPlot(
        sdata,
        output=cfg.paths.preprocess_adata,
    )

    fc.plot_shapes(
        sdata,
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
    sdata = fc.filter_on_size(
        sdata,
        min_size=cfg.allocate.min_size,
        max_size=cfg.allocate.max_size,
    )

    fc.plot_shapes(
        sdata,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        column=cfg.allocate.shape_size_column,
        cmap=cfg.allocate.shape_size_cmap,
        alpha=cfg.allocate.shape_size_alpha,
        output=cfg.paths.shape_size,
    )

    print("Start clustering")

    sdata = fc.clustering(
        sdata,
        pcs=cfg.allocate.pcs,
        neighbors=cfg.allocate.neighbors,
        cluster_resolution=cfg.allocate.cluster_resolution,
    )

    fc.clustering_plot(
        sdata,
        output=cfg.paths.cluster,
    )

    fc.plot_shapes(
        sdata,
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
    mg_dict, scoresper_cluster = fc.scoreGenes(
        sdata=sdata,
        path_marker_genes=cfg.dataset.markers,
        delimiter=cfg.annotate.delimiter,
        row_norm=cfg.annotate.row_norm,
        repl_columns=repl_columns,
        del_celltypes=del_celltypes,
    )

    if cfg.segmentation.voronoi_radius:
        shapes_layer = "expanded_cells" + str(cfg.segmentation.voronoi_radius)
    else:
        for key in sdata.shapes.keys():
            if "boundaries" in key:
                shapes_layer = key
                break

    fc.scoreGenesPlot(
        sdata=sdata,
        scoresper_cluster=scoresper_cluster,
        shapes_layer=shapes_layer,
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
    if "marker_genes" in cfg.visualize:
        sdata = fc.correct_marker_genes(
            sdata,
            celltype_correction_dict=cfg.visualize.marker_genes,
        )

    # Get arguments from cfg else None objects
    celltype_indexes = (
        cfg.visualize.celltype_indexes if "celltype_indexes" in cfg.visualize else None
    )
    colors = cfg.visualize.colors if "colors" in cfg.visualize else None

    # Check cluster cleanliness
    sdata, color_dict = fc.clustercleanliness(
        sdata,
        genes=list(mg_dict.keys()),
        gene_indexes=celltype_indexes,
        colors=colors,
    )

    if cfg.segmentation.voronoi_radius:
        shapes_layer = "expanded_cells" + str(cfg.segmentation.voronoi_radius)
    else:
        for key in sdata.shapes.keys():
            if "boundaries" in key:
                shapes_layer = key
                break

    fc.clustercleanlinessPlot(
        sdata=sdata,
        shapes_layer=shapes_layer,
        crd=cfg.segmentation.small_size_vis
        if cfg.segmentation.small_size_vis is not None
        else cfg.clean.small_size_vis,
        color_dict=color_dict,
        output=cfg.paths.cluster_cleanliness,
    )

    # calculate nhood enrichment
    sdata = fc.enrichment(sdata)
    fc.enrichment_plot(
        sdata,
        output=cfg.paths.nhood,
    )

    return sdata
