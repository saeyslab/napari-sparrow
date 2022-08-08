# this file acts as a robust starting point for launching hydra runs and multiruns
# can be run from any place


import hydra
import numpy as np
import pyrootutils
from omegaconf import DictConfig

from napari_spongepy import utils

log = utils.get_pylogger(__name__)

# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
# sets PROJECT_ROOT environment variable (used in `configs/paths/default.yaml`)
# loads environment variables from ".env" if exists
# adds root dir to the PYTHONPATH (so this file can be run from any place)
# https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def mask_to_polygons_layer(mask):
    import geopandas
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    import rasterio as features
    import shapely

    # https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    all_polygons = []
    all_values = []
    for shape, value in features.shapes(
        mask.astype(np.int16),
        mask=(mask > 0),
        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shapely.geometry.shape(shape))
        all_values.append(int(value))

    def color(r):
        return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))

    def border_color(r):
        return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)

    def linewidth(r):
        return 1 if r else 0.5

    def is_in_border(r):
        r = r.centroid
        if (r.x - border_margin < 0) or (r.x + border_margin > h):
            return True
        if (r.y - border_margin < 0) or (r.y + border_margin > w):
            return True
        return False

    polygons = geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)
    border_margin = 30
    w, h = mask.shape[0], mask.shape[1]
    polygons["border"] = polygons.geometry.map(is_in_border)
    polygons["border_color"] = polygons.border.map(border_color)
    polygons["linewidth"] = polygons.border.map(linewidth)
    polygons["color"] = polygons.border.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")
    polygons["x"] = polygons.centroid.map(lambda p: p.x)
    polygons["y"] = polygons.centroid.map(lambda p: p.y)
    polygons["size"] = polygons.area
    return polygons


@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="pipeline.yaml"
)
def main(cfg: DictConfig) -> None:

    # preprocessing
    import pipelineScripts as pl

    crd = [4500, 4600, 6500, 6700]
    log.info("Start preprocessing")
    img = pl.preprocessImage(
        path_image=cfg.dataset.image,
        size_tophat=45,
        small_size_vis=crd,
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
    return
    df = pl.allocate_genes_quick(cfg.dataset.coords, masks)

    coordinates = (
        df.groupby(["cells"]).mean().iloc[:, [0, 1]]
    )  # calculate the mean of the transcripts for every cell. Now based on transcripts, better on masks?
    # based on masks is present in the adata.obsm

    # cellCounts=df.groupby(['cells',3]).size().unstack(fill_value=0) #create a matrix based on counts
    # adata = AnnData(cellCounts[cellCounts.index!=0])
    coordinates.index = coordinates.index.map(str)
    # adata.obsm['spatial'] = coordinates[coordinates.index!='0']

    # polygonsF=polygons[np.isin(polygons.index.values,list(map(int,adata.obs.index.values)))]
    # polygonsF.index=list(map(str,polygonsF.index))
    # adata.obsm['polygons']=polygonsF
    # adata = pl.create_adata_quick(df, img, polygons)
    # adata = pl.preprocessAdata(adata, masks)


if __name__ == "__main__":
    main()
