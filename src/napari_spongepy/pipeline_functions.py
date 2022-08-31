from omegaconf import DictConfig
from skimage import io

from napari_spongepy import functions as fc
from napari_spongepy import utils

log = utils.get_pylogger(__name__)


def clean(cfg: DictConfig, results: dict) -> DictConfig:
    img = io.imread(cfg.dataset.image)

    # Perform tilingCorrection on whole image
    if cfg.clean.tilingCorrection:
        img_correct, flatfield = fc.tilingCorrection(img=img, device=cfg.clean.device)
        if "tiling_correction" in cfg.paths:
            log.info(f"Writing tiling plots to {cfg.paths.tiling_correction}")
            fc.tilingCorrectionPlot(
                img_correct, flatfield, img, cfg.paths.tiling_correction
            )
        img = img_correct

    # Image subset for faster processing
    if cfg.subset:
        subset = utils.parse_subset(cfg.subset)
        log.info(f"Subset is {subset}")
        img = img[subset]

    # Preprocess Image
    img_preprocess = fc.preprocessImage(
        img=img,
        size_tophat=cfg.clean.size_tophat,
        contrast_clip=cfg.clean.contrast_clip,
    )

    if "preprocess" in cfg.paths:
        log.info(f"Writing preprocess plots to {cfg.paths.preprocess}")
        fc.preprocessImagePlot(
            img_preprocess, img, cfg.clean.small_size_vis, cfg.paths.preprocess
        )

    results = {"preprocessimg": img_preprocess}

    return cfg, results


def segment(cfg: DictConfig, results: dict) -> DictConfig:
    import numpy as np

    img = results["preprocessimg"]

    masks, masks_i, polygons = fc.segmentation(
        img,
        cfg.device,
        cfg.segmentation.min_size,
        cfg.segmentation.flow_threshold,
        cfg.segmentation.diameter,
        cfg.segmentation.cellprob_threshold,
        cfg.segmentation.model_type,
        cfg.segmentation.channels,
    )

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

    if "masks" in cfg.paths:
        log.info(f"Writing masks to {cfg.paths.masks}")
        np.save(cfg.paths.masks, masks)
    results["segmentationmasks"] = masks

    return cfg, results


def allocate(cfg: DictConfig, results: dict) -> DictConfig:
    masks = results["segmentationmasks"]
    img = results["preprocessimg"]

    adata = fc.create_adata_quick(
        cfg.dataset.coords, img, masks, cfg.allocate.library_id
    )
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

    adata, adata_orig = fc.preprocessAdata(
        adata, masks, cfg.allocate.nuc_size_norm, cfg.allocate.n_comps
    )
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

    adata, _ = fc.filter_on_size(adata, cfg.allocate.min_size, cfg.allocate.max_size)
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

    fc.clustering(
        adata,
        cfg.allocate.pcs,
        cfg.allocate.neighbors,
        cfg.allocate.cluster_resolution,
        output=cfg.paths.score_genes,
    )
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
    adata = results["adata"]
    repl_columns = (
        cfg.annotate.repl_columns if "repl_columns" in cfg.annotate else dict()
    )
    del_genes = cfg.annotate.del_genes if "del_genes" in cfg.annotate else []
    mg_dict, _ = fc.scoreGenes(
        adata, cfg.dataset.markers, cfg.annotate.row_norm, repl_columns, del_genes
    )
    if "marker_genes" in cfg.annotate:
        adata = fc.correct_marker_genes(adata, cfg.annotate.marker_genes)
    results["adata"] = adata
    results["mg_dict"] = mg_dict

    return cfg, results


def visualize(cfg: DictConfig, results: dict) -> DictConfig:
    adata = results["adata"]
    mg_dict = results["mg_dict"]

    gene_indexes = (
        cfg.visualize.gene_indexes if "gene_indexes" in cfg.visualize else None
    )
    colors = cfg.visualize.colors if "colors" in cfg.visualize else None

    adata, color_dict = fc.clustercleanliness(
        adata, list(mg_dict.keys()), gene_indexes, colors
    )

    if "cluster_cleanliness" in cfg.paths:
        log.info(f"Writing cluster cleanliness plot to {cfg.paths.cluster_cleanliness}")
        fc.clustercleanlinessPlot(
            adata,
            cfg.visualize.crd,
            color_dict,
            output=cfg.paths.cluster_cleanliness,
        )

    adata = fc.enrichment(adata)
    if "nhood" in cfg.paths:
        fc.enrichment_plot(adata, cfg.paths.nhood)

    fc.save_data(adata, cfg.paths.geojson, cfg.paths.h5ad)
    log.info("Pipeline finished")

    return cfg, results
