import pathlib

import napari
import napari.layers
import napari.types
from magicgui import magic_factory

import napari_spongepy.utils as utils
from napari_spongepy import functions as fc

log = utils.get_pylogger(__name__)


@magic_factory(call_button="Allocate")
def allocate_widget(
    viewer: napari.Viewer,
    transcripts_file: pathlib.Path = pathlib.Path(""),
    min_size=500,
    pcs: int = 17,
    neighbors: int = 35,
    spot_size: int = 70,
    cluster_resolution: float = 0.8,
):
    ic = utils.get_ic()

    img = ic[utils.CLEAN].data.squeeze()
    masks = ic[utils.SEGMENT].data.squeeze()

    log.info(f"path is {transcripts_file}")

    adata = fc.create_adata_quick(str(transcripts_file), img, masks)
    adata, _ = fc.preprocessAdata(adata, masks)
    adata, _ = fc.filter_on_size(adata, min_size=min_size)
    fc.clustering(adata, pcs, neighbors, spot_size, cluster_resolution)

    log.info(f"adata is {adata}")


if __name__ == "__main__":
    from hydra import compose, initialize
    from hydra.core.hydra_config import HydraConfig

    import napari_spongepy.pipeline_functions as pf

    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                "+dataset=resolve_liver",
                "+segmentation=watershed",
                "dataset.image=${dataset.data_dir}/subset_20272_slide1_A1-1_DAPI.tiff",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

    results: dict = {}
    cfg, results = pf.clean(cfg, results)
    cfg, results = pf.segment(cfg, results)

    viewer = napari.Viewer()
    allocate_widget(viewer=viewer, transcripts_file=cfg.dataset.coords)
