""" This file contains the five pipeline steps that are used by the single pipeline.
Some steps consist of multiple substeps from the functions file. """

import squidpy.im as sq
from omegaconf import DictConfig
from skimage import io

from napari_sparrow import functions as fc
from napari_sparrow import utils

log = utils.get_pylogger(__name__)


def clean(cfg: DictConfig, results: dict) -> DictConfig:
    """Cleaning step, the first step of the pipeline, performs tilingCorrection and preprocessing of the image to improve image quality."""

    # Read in the image, load into ImageContainer
    img = io.imread(cfg.dataset.image)
    ic = sq.ImageContainer(img)

    # Image subset for faster processing
    left_corner, size = None, None
    if cfg.subset:
        left_corner, size = utils.parse_subset(cfg.subset)
        log.info(f"Subset is {str(cfg.subset)}")

    # Perform tilingCorrection on the whole image, corrects illumination and performs inpainting
    if cfg.clean.tilingCorrection:

        # Left_corner and size for subsetting the image
        ic_correct, flatfield = fc.tilingCorrection(
            ic, left_corner, size, cfg.clean.tile_size
        )

        # Write plot to given path if output is enabled
        if "tiling_correction" in cfg.paths:
            log.info(f"Writing tiling plots to {cfg.paths.tiling_correction}")
            fc.tilingCorrectionPlot(
                ic_correct.data.image.squeeze().to_numpy(),
                flatfield,
                img,
                cfg.paths.tiling_correction,
            )
        ic = ic_correct

    # Preprocess image, apply tophat filter if supplied and CLAHE contrast function
    ic_preprocess = fc.preprocessImage(
        img=ic,
        size_tophat=cfg.clean.size_tophat,
        contrast_clip=cfg.clean.contrast_clip,
    )

    # Write plot to given path if output is enabled
    if "preprocess" in cfg.paths:
        log.info(f"Writing preprocess plots to {cfg.paths.preprocess}")
        fc.preprocessImagePlot(
            ic_preprocess.data.image.squeeze().to_numpy(),
            img,
            cfg.clean.small_size_vis,
            cfg.paths.preprocess,
        )

    results = {"preprocessimg": ic_preprocess}

    return cfg, results


def segment(cfg: DictConfig, results: dict) -> DictConfig:
    """Segmentation step, the second step of the pipeline, performs cellpose segmentation and creates masks."""

    import numpy as np

    # Load image from previous step
    ic = results["preprocessimg"]
    img = ic.data.image.squeeze().to_numpy()

    # Perform segmentation
    masks, masks_i, polygons, ic = fc.segmentation(
        ic,
        cfg.device,
        cfg.segmentation.min_size,
        cfg.segmentation.flow_threshold,
        cfg.segmentation.diameter,
        cfg.segmentation.cellprob_threshold,
        cfg.segmentation.model_type,
        cfg.segmentation.channels,
    )

    # Write plot to given path if output is enabled
    if "segmentation" in cfg.paths:
        log.info(f"Writing segmentation plots to {cfg.paths.segmentation}")
        fc.segmentationPlot(
            img,
            masks_i,
            polygons,
            channels=cfg.segmentation.channels,
            small_size_vis=cfg.segmentation.small_size_vis,
            output=cfg.paths.segmentation,
        )

    # Write masks to file if output is enabled
    if "masks" in cfg.paths:
        log.info(f"Writing masks to {cfg.paths.masks}")
        np.save(cfg.paths.masks, masks)

    results["segmentationmasks"] = masks
    results["preprocessimg"] = ic

    return cfg, results


def allocate(cfg: DictConfig, results: dict) -> DictConfig:
    """Allocation step, the third step of the pipeline, creates the adata object from the mask and allocates the transcripts from the supplied file."""

    # Load results from previous steps
    masks = results["segmentationmasks"]
    img = results["preprocessimg"]

    # Create the adata object with from the masks and the transcripts
    adata = fc.create_adata_quick(
        cfg.dataset.coords, img, masks, cfg.allocate.library_id
    )

    # Write plots to given path if output is enabled
    if "polygons" in cfg.paths:
        log.info(f"Writing polygon plot to {cfg.paths.polygons}")
        fc.plot_shapes(
            adata,
            cfg.allocate.polygon_column or None,
            cfg.allocate.polygon_cmap,
            cfg.allocate.polygon_alpha,
            cfg.allocate.polygon_crd or None,
            output=cfg.paths.polygons,
        )

    # Perform normalization and remove all cells with less then 10 genes
    adata, adata_orig = fc.preprocessAdata(
        adata, masks, cfg.allocate.nuc_size_norm, cfg.allocate.n_comps
    )

    # Write plots to given path if output is enabled
    if "preprocess_adata" in cfg.paths:
        log.info(f"Writing preprocess_adata plot to {cfg.paths.preprocess_adata}")
        fc.preprocesAdataPlot(adata, adata_orig, output=cfg.paths.preprocess_adata)
    if "total_counts" in cfg.paths:
        log.info(f"Writing total count plot to {cfg.paths.total_counts}")
        fc.plot_shapes(
            adata,
            cfg.allocate.total_counts_column or None,
            cfg.allocate.total_counts_cmap,
            cfg.allocate.total_counts_alpha,
            cfg.allocate.total_counts_crd or None,
            output=cfg.paths.total_counts,
        )

    # Filter all cells based on size and distance
    adata, _ = fc.filter_on_size(adata, cfg.allocate.min_size, cfg.allocate.max_size)

    # Write plots to given path if output is enabled
    if "distance" in cfg.paths:
        log.info(f"Writing distance plot to {cfg.paths.distance}")
        fc.plot_shapes(
            adata,
            cfg.allocate.distance_column or None,
            cfg.allocate.distance_cmap,
            cfg.allocate.distance_alpha,
            cfg.allocate.distance_crd or None,
            output=cfg.paths.distance,
        )

    # Extract segmentation features, area and mean_intensity
    adata = fc.extract(img, adata)

    # Perform neighborhood analysis and leiden clustering
    adata = fc.clustering(
        adata,
        cfg.allocate.pcs,
        cfg.allocate.neighbors,
        cfg.allocate.cluster_resolution,
    )

    # Write plot to given path if output is enabled
    if "cluster" in cfg.paths:
        log.info(f"Writing clustering plots to {cfg.paths.cluster}")
        fc.clustering_plot(adata, output=cfg.paths.cluster)
    if "leiden" in cfg.paths:
        log.info(f"Writing leiden plot to {cfg.paths.leiden}")
        fc.plot_shapes(
            adata,
            cfg.allocate.leiden_column or None,
            cfg.allocate.leiden_cmap,
            cfg.allocate.leiden_alpha,
            cfg.allocate.leiden_crd or None,
            output=cfg.paths.leiden,
        )

    results["adata"] = adata

    return cfg, results


def annotate(cfg: DictConfig, results: dict) -> DictConfig:
    """Annotation step, the fourth step of the pipeline, annotates the cells with celltypes based on the marker genes file."""

    # Load data from previous step
    adata = results["adata"]

    # Get arguments from cfg else empty objects
    repl_columns = (
        cfg.annotate.repl_columns if "repl_columns" in cfg.annotate else dict()
    )
    del_genes = cfg.annotate.del_genes if "del_genes" in cfg.annotate else []

    # Load marker genes, replace columns with different name, delete genes from list
    mg_dict, scoresper_cluster = fc.scoreGenes(
        adata, cfg.dataset.markers, cfg.annotate.row_norm, repl_columns, del_genes
    )

    # Write plot to given path if output is enabled
    if "score_genes" in cfg.paths:
        fc.scoreGenesPlot(adata, scoresper_cluster, output=cfg.paths.score_genes)

    # Perform correction for genes that occur in all cells and are overexpressed
    if "marker_genes" in cfg.annotate:
        adata = fc.correct_marker_genes(adata, cfg.annotate.marker_genes)

    results["adata"] = adata
    results["mg_dict"] = mg_dict

    return cfg, results


def visualize(cfg: DictConfig, results: dict) -> DictConfig:
    """Visualisation step, the fifth and final step of the pipeline, checks the cluster cleanliness and performs nhood enrichement before saving the data as geojson and h5ad files."""

    # Load data from previous step
    adata = results["adata"]
    mg_dict = results["mg_dict"]

    # Get arguments from cfg else None objects
    gene_indexes = (
        cfg.visualize.gene_indexes if "gene_indexes" in cfg.visualize else None
    )
    colors = cfg.visualize.colors if "colors" in cfg.visualize else None

    # Check cluster cleanliness
    adata, color_dict = fc.clustercleanliness(
        adata, list(mg_dict.keys()), gene_indexes, colors
    )

    # Write plot to given path if output is enabled
    if "cluster_cleanliness" in cfg.paths:
        log.info(f"Writing cluster cleanliness plot to {cfg.paths.cluster_cleanliness}")
        fc.clustercleanlinessPlot(
            adata,
            cfg.visualize.crd,
            color_dict,
            output=cfg.paths.cluster_cleanliness,
        )
    # Calculate nhood enrichement
    adata = fc.enrichment(adata)
    if "nhood" in cfg.paths:
        fc.enrichment_plot(adata, cfg.paths.nhood)

    # Save polygons to geojson and adata to h5ad files
    fc.save_data(adata, cfg.paths.geojson, cfg.paths.h5ad)

    log.info("Pipeline finished")
    return cfg, results
