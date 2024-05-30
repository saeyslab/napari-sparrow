
def pixie_example():
    from datasets import load_dataset
    import spatialdata as sd
    from pathlib import Path
    from dask_image.imread import imread
    import dask.array as da
    from loguru import logger


    # ['segment_image_data', 'cluster_pixels', 'cluster_cells', 'post_clustering', 'fiber_segmentation', 'LDA_preprocessing', 'LDA_training_inference', 'neighborhood_analysis', 'pairwise_spatial_enrichment', 'ome_tiff', 'ez_seg_data']
    dataset = load_dataset("angelolab/ark_example", 'cluster_cells', trust_remote_code=True)
    logger.info(dataset)

    # get location of images
    path = dataset['cluster_cells']['image_data'][0]
    logger.info(path)
    p = Path(path) / 'image_data'
    assert p.exists(), p
    # get fov1
    p = p / 'fov1'

    channel_paths = list(p.iterdir())
    channel_names = [c.stem for c in channel_paths]
    logger.info(len(channel_paths))
    logger.info(channel_names)

    imgs = [imread(img) for img in channel_paths]
    stack = da.stack(imgs).squeeze()

    logger.info(stack.shape)
    
    sdata = sd.SpatialData(
        images={
            "fov1": sd.models.Image2DModel.parse(
                data=stack,
                c_coords=channel_names
            )
        }
    )

    logger.info(sdata)
    # 
    return dataset


if __name__ == "__main__":
    pixie_example()