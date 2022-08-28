# %load_ext autoreload
# %autoreload 2
from typing import List, Optional, Tuple

# %matplotlib widget
import cv2
import geopandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scanpy as sc
import seaborn as sns
import shapely
import torch
from anndata import AnnData
from basicpy import BaSiC
from cellpose import models
from rasterio import features
from scipy import ndimage


def BasiCCorrection(
    img: np.ndarray, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    "This function corrects for the tiling effect that occurs in RESOLVE data"
    # create the tiles
    tiles = np.array(
        [
            img[i : i + 2144, j : j + 2144]
            for i in range(0, img.shape[0], 2144)
            for j in range(0, img.shape[1], 2144)
        ]
    )

    # measure the filters
    device = torch.device(device)
    torch.cuda.set_device(device)

    basic = BaSiC(epsilon=1e-06, device="cpu" if device == "cpu" else "gpu")

    device = torch.device(device)
    torch.cuda.set_device(device)

    basic.fit(tiles)
    flatfield = basic.flatfield
    tiles_corrected = basic.transform(tiles)

    # stitch the tiles back together
    i_new = np.block(
        [
            list(tiles_corrected[i : i + (img.shape[1] // 2144)])
            for i in range(0, len(tiles_corrected), img.shape[1] // 2144)
        ]
    )

    return i_new.astype(np.uint16), flatfield


def BasiCCorrectionPlot(img: np.ndarray, flatfield, img_orig: np.ndarray) -> None:
    plt.imshow(flatfield, cmap="gray")
    plt.title("Correction performed per tile")

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Corrected image")
    ax[1].imshow(img_orig, cmap="gray")
    ax[1].set_title("Original image")


def preprocessImage(
    img: np.ndarray,
    contrast_clip: float = 2.5,
    size_tophat: int = None,
) -> np.ndarray:
    "This function performs the prprocessing of an image. If the path_image i provided, the image is read from the path."
    "If the image img itself is provided, this image will be used."
    "Contrast_clip indiactes the input to the create_CLAHE function for histogram equalization"
    "size_tophat indicates the tophat filter size. If no tophat lfiter size is given, no tophat filter is executes. The recommendable size is 45?-."
    "Small_size_vis indicates the coordinates of an optional zoom in plot to check the processing better."

    # perform inpainting
    img = cv2.inpaint(img, (img == 0).astype(np.uint8), 55, cv2.INPAINT_NS)

    # tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(img, size_tophat)
        max_of_min_t = ndimage.maximum_filter(minimum_t, size_tophat)
        img -= max_of_min_t

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img


def preprocessImagePlot(
    img: np.ndarray,
    img_orig: np.ndarray,
    small_size_vis: List[int] = None,
) -> None:
    # plot_result
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Corrected image")
    ax[1].imshow(img_orig, cmap="gray")
    ax[1].set_title("Original image")

    # plot small part of image
    if small_size_vis is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax[0].set_title("Corrected image")
        ax[1].imshow(
            img_orig[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax[1].set_title("Original image")


def segmentation(
    img: np.ndarray,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    mask_threshold: int = 0,
    small_size_vis: List[int] = None,
    model_type: str = "nuclei",
    channels: np.ndarray = np.array([0, 0]),
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Optional[List[int]], geopandas.GeoDataFrame
]:
    "This function segments the data, using the cellpose algorithm, and plots the outcome"
    "img is the input image, showing the DAPI Staining, you can define your device by setting the device parameter"
    "min_size indicates the minimal amount of pixels in a mask (I assume)"
    "The lfow_threshold indicates someting about the shape of the masks, if you increase it, more masks with less orund shapes will be accepted"
    "The diameter is a very important parameter to estimate. In the best case, you estimate it yourself, it indicates the mean expected diameter of your dataset."
    "If you put None in diameter, them odel will estimate is herself."
    "mask_threshold indicates how many of the possible masks are kept. MAking it smaller (up to -6), will give you more masks, bigger is less masks. "
    "When an RGB image is given a input, the R channel is expected to have the nuclei, and the blue channel the membranes"
    "When whole cell segmentation needs to be performed, model_type=cyto, otherwise, model_type=nuclei"

    model = models.Cellpose(device=torch.device(device), model_type=model_type)

    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=mask_threshold,
    )
    mask_i = np.ma.masked_where(masks == 0, masks)
    # i_masked = np.ma.masked_where(img < 500, img)

    # create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    # polygons["border"] = polygons.geometry.map(is_in_border)
    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")

    return masks, mask_i, channels, small_size_vis, polygons


def segmentationPlot(
    img: np.ndarray,
    mask_i: np.ndarray,
    channels: np.ndarray,
    small_size_vis: List[int],
    polygons: geopandas.GeoDataFrame,
) -> None:
    if sum(channels) != 0:
        img = img[0, :, :]  # select correct image

    if small_size_vis is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )

        ax[1].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax[1].imshow(
            mask_i[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="jet",
        )
        plt.show()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # ax[0].imshow(masks[0:3000,8000:10000],cmap='jet')
        ax[0].imshow(img, cmap="gray")
        # ax[0].imshow(masks,cmap='jet')

        ax[1].imshow(img, cmap="gray")
        polygons.plot(
            ax=ax[1],
            edgecolor="white",
            linewidth=polygons.linewidth,
            alpha=0.5,
            legend=True,
            color="red",
        )
        # ax[1].imshow(masksI,cmap='jet')
        plt.show()


def mask_to_polygons_layer(mask: np.ndarray) -> geopandas.GeoDataFrame:
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

    return geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)


def color(_) -> matplotlib.colors.Colormap:
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    return 1 if r else 0.5


def is_in_border(r, h, w, border_margin):
    r = r.centroid
    if (r.x - border_margin < 0) or (r.x + border_margin > h):
        return True
    if (r.y - border_margin < 0) or (r.y + border_margin > w):
        return True
    return False


def create_adata_quick(
    path: str, img: np.ndarray, masks: np.ndarray, library_id: str = "melanoma"
) -> AnnData:

    # create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    # polygons["border"] = polygons.geometry.map(is_in_border)
    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")

    # allocate the transcripts
    df = pd.read_csv(path, delimiter="\t", header=None)
    df = df[(df[1] < masks.shape[0]) & (df[0] < masks.shape[1])]
    df["cells"] = masks[df[1].values, df[0].values]

    coordinates = df.groupby(["cells"]).mean().iloc[:, [0, 1]]
    # calculate the mean of the transcripts for every cell. Now based on transcripts, better on masks?
    # based on masks is present in the adata.obsm
    # create the anndata object
    cell_counts = (
        df.groupby(["cells", 3]).size().unstack(fill_value=0)
    )  # create a matrix based on counts
    adata = AnnData(cell_counts[cell_counts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"]

    # add the polygons to the anndata object
    polygons_f = polygons[
        np.isin(polygons.index.values, list(map(int, adata.obs.index.values)))
    ]
    polygons_f.index = list(map(str, polygons_f.index))
    adata.obsm["polygons"] = polygons_f

    # add the figure to the anndata
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {"hires": img}
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }

    return adata


def plot_shapes(
    adata: AnnData,
    column: str = None,
    cmap: str = "magma",
    alpha: float = 0.5,
    crd: List[int] = None,
) -> None:
    "This function plots the anndata on the shapes of the cells, but it does not do it smartly."

    if column is not None:
        if column + "_colors" in adata.uns:
            print("Using the colormap defined in the anndata object")
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "new_map",
                adata.uns[column + "_colors"],
                N=len(adata.uns[column + "_colors"]),
            )

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(
            adata.uns["spatial"]["melanoma"]["images"]["hires"],
            cmap="gray",
        )
        adata.obsm["polygons"].plot(
            ax=ax,
            column=adata.obs[column],
            edgecolor="white",
            linewidth=adata.obsm["polygons"].linewidth,
            alpha=alpha,
            legend=True,
            cmap=cmap,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(
            adata.uns["spatial"]["melanoma"]["images"]["hires"],
            cmap="gray",
        )
        adata.obsm["polygons"].plot(
            ax=ax,
            edgecolor="white",
            linewidth=adata.obsm["polygons"].linewidth,
            alpha=alpha,
            legend=True,
            color="blue",
        )

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # ax.legend(bbox_to_anchor=(1.1, 1.05))
    if crd is not None:
        ax.set_xlim(crd[0], crd[1])
        ax.set_ylim(crd[2], crd[3])
    # ax[1].imshow(I,cmap='gray',)


def preprocessAdata(
    adata: AnnData, mask: np.ndarray, nuc_size_norm: bool = True, n_comps: int = 50
) -> Tuple[AnnData, AnnData]:
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[2, 5])
    adata_orig = adata
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.raw = adata

    # nucleusSizeNormalization
    if nuc_size_norm:
        _, counts = np.unique(mask, return_counts=True)
        adata.obs["nucleusSize"] = [counts[int(index)] for index in adata.obs.index]
        adata.X = (adata.X.T / adata.obs.nucleusSize.values).T

        # sc.pp.normalize_total(adata) #This no
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps)
    sc.pl.pca(adata, color="total_counts")
    # sc.pl.pca_variance_ratio(adata,n_pcs=50) #lets take 6,10 or 12
    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )

    return adata, adata_orig


def preprocesAdataPlot(adata: AnnData, adata_orig: AnnData) -> None:
    _, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.distplot(adata_orig.obs["total_counts"], kde=False, ax=axs[0])
    sns.distplot(adata_orig.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])

    plt.scatter(adata.obs["nucleusSize"], adata.obs["total_counts"])
    plt.title = "cellsize vs cellcount"
    plt.show()


def filter_on_size(
    adata: AnnData, min_size: int = 100, max_size: int = 100000
) -> Tuple[AnnData, int]:
    start = adata.shape[0]
    adata.obsm["polygons"]["X"] = adata.obsm["polygons"].centroid.x
    adata.obsm["polygons"]["Y"] = adata.obsm["polygons"].centroid.y
    adata.obs["distance"] = np.sqrt(
        np.square(adata.obsm["polygons"]["X"] - adata.obsm["spatial"][0])
        + np.square(adata.obsm["polygons"]["Y"] - adata.obsm["spatial"][1])
    )

    adata = adata[adata.obs["nucleusSize"] < max_size, :]
    adata = adata[adata.obs["nucleusSize"] > min_size, :]
    adata = adata[adata.obs["distance"] < 70, :]

    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )
    filtered = start - adata.shape[0]

    return adata, filtered


def clustering(
    adata: AnnData,
    pcs: int,
    neighbors: int,
    spot_size: int = 70,
    cluster_resolution: float = 0.8,
) -> Tuple[AnnData, pd.DataFrame]:
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.umap(adata)
    # sc.pl.umap(adata, color=['Folr2','Glul','Sox9','Cd9']) #total counts doesn't matter that much
    sc.tl.leiden(adata, resolution=cluster_resolution)
    sc.pl.umap(adata, color=["leiden"])
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False)

    return adata


def scoreGenesLiver(
    adata: AnnData, path_marker_genes: str, row_norm: bool = False
) -> Tuple[dict, pd.DataFrame]:
    df_markers = pd.read_csv(path_marker_genes, index_col=0)
    df_markers.columns = df_markers.columns.str.replace("Tot_Score_", "")
    df_markers.columns = df_markers.columns.str.replace("uppfer", "upffer")
    genes_dict = {}
    for i in df_markers:
        genes = []
        for row, value in enumerate(df_markers[i]):
            if value > 0:
                genes.append(df_markers.index[row])
                # print(df_markers.index[row])
        genes_dict[i] = genes

    for key, value in genes_dict.items():
        sc.tl.score_genes(adata, value, score_name=key)

    # scoresper_cluster = adata.obs[[col for col in adata.obs if col.startswith('Tot')]] #very specific to this dataset
    scoresper_cluster = adata.obs[
        [col for col in adata.obs if col in df_markers.columns]
    ]
    if row_norm:
        row_norm = scoresper_cluster.sub(
            scoresper_cluster.mean(axis=1).values, axis="rows"
        ).div(
            scoresper_cluster.std(axis=1).values, axis="rows"
        )  # row normalization
        # Row normalization is just there for visualization purposes, to make sure we are not overdoing it

        adata.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])
    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    adata.obs["Cleanliness"] = scores.values
    adata.obs["maxScores"] = scoresper_cluster.idxmax(axis=1)

    return genes_dict, scoresper_cluster


def scoreGenesLiverPlot(adata: AnnData, scoresper_cluster: pd.DataFrame) -> None:
    sc.pl.umap(adata, color=["Cleanliness", "maxScores"])
    sc.pl.umap(adata, color=["leiden", "maxScores"])
    # fig,ax =plt.subplots(1,1,figsize=(20,10))

    # sc.pl.spatial(adata,color='maxScores',spot_size=70,show=False,cmap='magma',alpha=1,title='AnnotationScores',ax=ax)
    # fig,ax =plt.subplots(1,1,figsize=(20,10)) sc.pl.spatial(adata,color='Cleanliness',spot_size=70,show=False,
    # cmap='magma',alpha=1,title='Cleanliness of data',ax=ax)

    plot_shapes(adata, column="maxScores")
    plot_shapes(adata, column="Cleanliness")

    sc.pl.heatmap(adata, var_names=scoresper_cluster.columns.values, groupby="leiden")
    sc.pl.heatmap(
        adata[
            adata.obs.leiden.isin(
                ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
            )
        ],
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
    )


def clustercleanliness(
    adata: AnnData,
    img: np.ndarray,
    genes: List[str],
    crop_coord: List[int] = [0, 2000, 0, 2000],
    liver: bool = False,
) -> None:
    celltypes = np.array(sorted(genes))

    # The coloring doesn't work yet for non-liver samples, but is easily adaptable, by just not defining a colormap
    # anywhere
    adata.obs["maxScores"] = adata.obs[
        [col for col in adata.obs if col in celltypes]
    ].idxmax(axis=1)
    adata.obs.maxScores = adata.obs.maxScores.astype("category")

    if liver:
        other_immune_cells = celltypes[1, 4, 7, 8, 9, 10, 11, 12, 13, 17]
        vein = celltypes[15, 18]
        adata.obs["maxScoresSave"] = adata.obs.maxScores

        adata.obs.maxScores = adata.obs.maxScores.cat.add_categories(
            ["Other_ImmuneCells"]
        )
        for i, val in enumerate(adata.obs.maxScores):
            if val in other_immune_cells:
                adata.obs.maxScores[i] = "Other_ImmuneCells"
        adata.obs.maxScores = adata.obs.maxScores.cat.add_categories(["vein_EC45"])

        for i, val in enumerate(adata.obs.maxScores):
            if val in vein:
                adata.obs.maxScores[i] = "vein_EC45"
        adata.obs.maxScoresSave = adata.obs.maxScoresSave.cat.add_categories(
            ["vein_EC45"]
        )

        for i in range(0, len(adata.obs.maxScores)):
            if adata.obs.maxScoresSave[i] in vein:
                adata.obs.maxScoresSave[i] = "vein_EC45"
        for i in other_immune_cells:
            if i in adata.obs.maxScores.cat.categories:
                adata.obs.maxScores = adata.obs.maxScores.cat.remove_categories(i)
        adata.obs.maxScores = adata.obs.maxScores.cat.remove_categories(vein)
        adata.obs.maxScoresSave = adata.obs.maxScoresSave.cat.remove_categories(vein)
        # fix the coloring

        # adata.uns['maxScores_colors']=np.append(adata.uns['maxScores_colors'],['#ff7f0e','#ad494a'])
        colors = [
            "#914d22",
            "#c61b84",
            "#ea579f",
            "#5da6db",
            "#fbb05f",
            "#d46f6c",
            "#E45466",
            "#a31a2a",
            "#929591",
            "#cc7722",
        ]
        adata.uns["maxScores_colors"] = colors

        color_dict = dict(
            zip(list(adata.obs.maxScores.cat.categories), adata.uns["maxScores_colors"])
        )
        for i, name in enumerate(color_dict.keys()):
            color_dict[name] = colors[i]

        colors_i = [
            "#191919",
            "#a3d7ba",
            "#702963",
            "#a4daf3",
            "#4a6e34",
            "#b4b5b5",
            "#3ab04a",
            "#893a86",
            "#bf00ff",
            "#9c7eba",
        ]
        for i in range(0, 10):
            color_dict[other_immune_cells[i]] = colors_i[i]

    # create the plots

    stacked = (
        adata.obs.groupby(["leiden", "maxScores"], as_index=False)
        .size()
        .pivot("leiden", "maxScores")
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(adata.obs.maxScores.cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if liver:
        stacked_norm.plot(
            kind="bar", stacked=True, ax=fig.gca(), color=color_dict
        )  # .legend(loc='lower left')
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
        # ax.axis('off')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    # plt.title('Cluster purity based on marker gene lists',fontsize='xx-large')
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")
    plt.show()

    plot_shapes(adata, column="maxScores", alpha=0.8)
    plot_shapes(adata, column="maxScores", crd=crop_coord, alpha=0.8)

    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(adata, color=["maxScores"], ax=ax, size=60, show=False)
    ax.axis("off")
    # plt.title('UMAP colored by annotation based on celltype ',fontsize='xx-large')
    plt.show()
