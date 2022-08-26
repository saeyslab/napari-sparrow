import pandas as pd
import squidpy as sq
from omegaconf import DictConfig
from skimage import io

from napari_spongepy import functions as fc
from napari_spongepy import utils

log = utils.get_pylogger(__name__)


def clean(cfg: DictConfig, results: dict) -> DictConfig:
    # Image subset for realtime checking
    if cfg.subset:
        subset = utils.parse_subset(cfg.subset)
        log.info(f"Subset is {subset}")
        img = io.imread(cfg.dataset.image)[subset]
    else:
        img = io.imread(cfg.dataset.image)

    # Perform BaSiCCorrection
    if cfg.clean.basic_correction:
        img, _ = fc.BasiCCorrection(img=img, device=cfg.clean.device)

    # Preprocess Image
    img, _ = fc.preprocessImage(
        img=img,
        size_tophat=cfg.clean.size_tophat,
        contrast_clip=cfg.clean.contrast_clip,
    )
    results = {"preprocessimg": img}

    return cfg, results


def segment(cfg: DictConfig, results: dict) -> DictConfig:
    import numpy as np

    img = results["preprocessimg"]

    masks, _, _, _, _ = fc.segmentation(
        img,
        cfg.segmentation.device,
        cfg.segmentation.min_size,
        cfg.segmentation.flow_threshold,
        cfg.segmentation.diameter,
        cfg.segmentation.cellprob_threshold,
    )

    if cfg.paths.masks:
        log.info(f"Writing masks to {cfg.paths.masks}")
        np.save(cfg.paths.masks, masks)
    results["segmentationmasks"] = masks

    return cfg, results


def allocate(cfg: DictConfig, results: dict) -> DictConfig:
    masks = results["segmentationmasks"]
    img = results["preprocessimg"]

    log.info(f"path is {cfg.dataset.coords}")
    adata = fc.create_adata_quick(cfg.dataset.coords, img, masks)
    adata, _ = fc.preprocessAdata(adata, masks)
    adata, _ = fc.filter_on_size(adata, min_size=500)
    fc.clustering(adata, 17, 35)

    results["adata"] = adata

    return cfg, results


def annotate(cfg: DictConfig, results: dict) -> DictConfig:
    adata = results["adata"]
    mg_dict, _ = fc.scoreGenesLiver(adata, cfg.dataset.markers)
    results["adata"] = adata
    results["mg_dict"] = mg_dict

    return cfg, results


def visualize(cfg: DictConfig, results: dict) -> DictConfig:
    adata = results["adata"]
    img = results["preprocessimg"]
    mg_dict = results["mg_dict"]
    crd = [2000, 4000, 3000, 5000]

    adata.obs["Hep"] = (adata.obs["Hepatocytes"] > 5.6).astype(int)

    for i in range(0, len(adata.obs)):
        if adata.obs["Hepatocytes"].iloc[i] < 5.6:
            adata.obs["Hepatocytes"].iloc[i] = adata.obs["Hepatocytes"].iloc[i] / 7

    fc.clustercleanliness(
        adata, img, genes=list(mg_dict.keys()), crop_coord=crd, liver=True
    )

    adata.raw.var.index.names = ["genes"]
    adata.var.index.names = ["genes"]
    adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="maxScores")

    del adata.obsm["polygons"]["color"]
    adata.obsm["polygons"]["geometry"].to_file(cfg.paths.geojson, driver="GeoJSON")

    adata.obsm["polygons"] = pd.DataFrame(
        {
            "linewidth": adata.obsm["polygons"]["linewidth"],
            "X": adata.obsm["polygons"]["X"],
            "Y": adata.obsm["polygons"]["Y"],
        }
    )
    adata.write(cfg.paths.h5ad)

    log.info("Pipeline finished")

    return cfg, results
