import hydra
import pandas as pd
import squidpy as sq
from omegaconf import DictConfig

from napari_spongepy import functions as fc
from napari_spongepy import utils

log = utils.get_pylogger(__name__)


def clean(cfg: DictConfig) -> DictConfig:
    # Perform BaSiCCorrection
    img, _, _ = fc.BasiCCorrection(img=cfg.dataset.image)

    # Preprocess Image
    img, _ = fc.preprocessImage(img=img, size_tophat=45, contrast_clip=3.5)
    cfg.result.preprocessimg = img

    return cfg


def segment(cfg: DictConfig) -> DictConfig:
    import numpy as np
    from squidpy.im import ImageContainer

    from napari_spongepy import utils

    ic, img = None, None

    if cfg.dataset.dtype == "xarray":
        # TODO support preprocessing for zarr datasets
        ic = ImageContainer(cfg.dataset.data_dir)
        print(ic)

        subset = cfg.subset
        if subset:
            subset = utils.parse_subset(subset)
            log.info(f"Subset is {subset}")
        # imports can be nested inside @hydra.main to optimize tab completion
        # https://github.com/facebookresearch/hydra/issues/934
        from napari_spongepy._segmentation_widget import _segmentation_worker

        if cfg.segmentation.get("method"):
            method = cfg.segmentation.method
        else:
            method = hydra.utils.instantiate(cfg.segmentation)

        worker = _segmentation_worker(
            ic,
            method=method,
            subset=subset,
            # TODO smarter selection of the z projection method
            reduce_z=3,
            reduce_c=3,
            # small chunks needed if subset is used
        )
    else:
        # crd = [4500, 4600, 6500, 6700]
        log.info("Start preprocessing")
        img, _ = fc.preprocessImage(
            path_image=cfg.dataset.image,
            size_tophat=45,
            contrast_clip=3.5,
        )
        # masks=pl.segmentation(img,device='mps',mask_threshold=-1,small_size_vis=crd,flow_threshold=0.7,min_size=1000)

        subset = cfg.subset
        if subset:
            subset = utils.parse_subset(subset)
            log.info(f"Subset is {subset}")
        # imports can be nested inside @hydra.main to optimize tab completion
        # https://github.com/facebookresearch/hydra/issues/934
        from napari_spongepy._segmentation_widget import _segmentation_worker

        if cfg.segmentation.get("method"):
            method = cfg.segmentation.method
        else:
            method = hydra.utils.instantiate(cfg.segmentation)

        worker = _segmentation_worker(
            img,
            method=method,
            subset=subset,
            # small chunks needed if subset is used
        )
    log.info("Start segmentation")
    [masks, _] = worker.work()
    log.info(masks.shape)

    # polygons = mask_to_polygons_layer(masks)
    if cfg.paths.masks:
        log.info(f"Writing masks to {cfg.paths.masks}")
        np.save(cfg.paths.masks, masks)
    cfg.result.segmentationmasks = masks

    return cfg


def allocate(cfg: DictConfig) -> DictConfig:
    masks = cfg.result.segmentationmasks
    adata = fc.create_adata_quick(cfg.dataset.coords, cfg.result.preprocessimg, masks)
    adata, _ = fc.preprocessAdata(adata, masks)
    adata, _ = fc.filter_on_size(adata, min_size=500)
    fc.clustering(adata, 17, 35)

    cfg.result.adata = adata

    return cfg


def annotate(cfg: DictConfig) -> DictConfig:
    adata = cfg.result.adata
    _, _ = fc.scoreGenesLiver(adata, cfg.dataset.markers)
    cfg.result.adata = adata

    return cfg


def visualize(cfg: DictConfig) -> DictConfig:
    adata = cfg.result.adata

    adata.raw.var.index.names = ["genes"]
    adata.var.index.names = ["genes"]
    adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="maxScores")
    sq.pl.nhood_enrichment(adata, cluster_key="maxScores", method="ward")

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

    return cfg
