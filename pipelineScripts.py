import time

#%matplotlib widget
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pipelineScripts as pl
import pybasic
import scanpy as sc
import scipy
import seaborn as sns
import torch
from anndata import AnnData
from cellpose import io, models
from scipy import ndimage
from skimage import io
from tqdm.notebook import tqdm


def BasiCCorrection(path_image=None, I=None):

    if I is None:
        I = io.imread(path_image)
    # create the tiles

    Tiles = []
    for i in range(0, int(I.shape[0] / 2144)):  # over the rows
        for j in range(0, int(I.shape[1] / 2144)):
            Temp = I[i * 2144 : (i + 1) * 2144, j * 2144 : (j + 1) * 2144]
            Tiles.append(Temp)
    # measure the filters
    flatfield = pybasic.basic(Tiles, darkfield=False, verbosity=False)

    tiles_corrected = pybasic.correct_illumination(
        images_list=Tiles, flatfield=flatfield[0]
    )  # , darkfield = darkfield)

    # stitch the tiles back together
    Inew = np.zeros(I.shape)
    k = 0
    for i in range(0, int(I.shape[0] / 2144)):  # over the rows
        for j in range(0, int(I.shape[1] / 2144)):
            Inew[
                i * 2144 : (i + 1) * 2144, j * 2144 : (j + 1) * 2144
            ] = tiles_corrected[k]
            k = k + 1

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(Inew, cmap="gray")
    ax[1].imshow(I, cmap="gray")

    return Inew.astype(np.uint16)


def preprocessImage(
    I=None, path_image=None, contrast_clip=2.5, size_tophat=None, small_size_vis=None
):
    "This function performs the prprocessing of an image. If the path_image i provided, the image is read from the path."
    "If the image I itself is provided, this image will be used."
    "Contrast_clip indiactes the input to the create_CLAHE function for histogram equalization"
    "size_tophat indicates the tophat filter size. If no tophat lfiter size is given, no tophat filter is executes. The recommendable size is 45?-."
    "Small_size_vis indicates the coordinates of an optional zoom in plot to check the processing better."
    t0 = time.time()
    # Read in image
    if I is None:
        I = io.imread(path_image)
    Iorig = I

    # mask black lines
    maskLines = np.where(I == 0)  # find the location of the lines
    mask = np.zeros(I.shape, dtype=np.uint8)
    mask[maskLines[0], maskLines[1]] = 1  # put one values in the correct position

    # perform inpainting
    res_NS = cv2.inpaint(I, mask, 15, cv2.INPAINT_NS)
    I = res_NS

    # tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(I, size_tophat)
        orig_sub_min = I - minimum_t
        I = orig_sub_min

    # enhance contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    I = clahe.apply(I)

    # plot_result
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(Iorig, cmap="gray")
    ax[1].imshow(I, cmap="gray")

    # plot small part of image
    if small_size_vis is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(
            Iorig[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax[1].imshow(
            I[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
    print(time.time() - t0)
    return I


def segmentation(
    I,
    device="cpu",
    min_size=80,
    flow_threshold=0.6,
    diameter=55,
    mask_threshold=0,
    small_size_vis=None,
):
    "This function segments the data, using the cellpose algorithm, and plots the outcome"
    "I is the input image, showing the DAPI Staining, you can define your device by setting the device parameter"
    "min_size indicates the minimal amount of pixels in a mask (I assume)"
    "The lfow_threshold indicates someting about the shape of the masks, if you increase it, more masks with less orund shapes will be accepted"
    "The diameter is a very important parameter to estimate. In the best case, you estimate it yourself, it indicates the mean expected diameter of your dataset."
    "If you put None in diameter, them odel will estimate is herself."
    "mask_threshold indicates how many of the possible masks are kept. MAking it smaller (up to -6), will give you more masks, bigger is less masks. "
    t0 = time.time()
    device = torch.device(device)  # GPU 4 is your GPU
    model = models.Cellpose(gpu=device, model_type="nuclei")
    channels = np.array([0, 0])

    masks, flows, styles, diams = model.eval(
        I,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=mask_threshold,
    )
    masksI = np.ma.masked_where(masks == 0, masks)
    Imasked = np.ma.masked_where(I < 500, I)

    # visualization

    if small_size_vis is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(
            I[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )

        ax[1].imshow(
            I[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax[1].imshow(
            masksI[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="jet",
        )
        plt.show()
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        # ax[0].imshow(masks[0:3000,8000:10000],cmap='jet')
        ax[0].imshow(I, cmap="gray")
        # ax[0].imshow(masks,cmap='jet')

        ax[1].imshow(I, cmap="gray")
        ax[1].imshow(masksI, cmap="jet")
        plt.show()
    print(time.time() - t0)
    return masks


def allocate_genes_quick(path, Polygons):
    # read in data
    df = pd.read_csv(path, delimiter="\t", header=None)
    df["cells"] = mask[df[1].values, df[0].values]
    return df


def allocate_genes(path, mask):
    # read in data
    df = pd.read_csv(path, delimiter="\t", header=None)

    # createcKDTree to speed up the process
    nonzero_masks = np.transpose(np.nonzero(mask))
    maskTree = scipy.spatial.cKDTree(nonzero_masks, leafsize=100)
    coordinates_genes = df.iloc[:, [0, 1]]

    # find the cell for every location
    coordinates_genes = df.iloc[:, [1, 0]]  # change the order to y,x
    cells = []
    distances = []
    t0 = time.time()
    for index, coordinates in tqdm(
        coordinates_genes.iterrows(), total=coordinates_genes.shape[0]
    ):
        outcome = maskTree.query(
            coordinates, k=1
        )  # here you can put a cut off for your distance: can't be further than x.

        # now you are only looking at one nearest neighbor, two can be interesting.
        coordinatesC = nonzero_masks[outcome[1]]
        distances.append(outcome[0])
        cell = mask[coordinatesC[0], coordinatesC[1]]
        cells.append(cell)
    t1 = time.time()
    print(t1 - t0)
    return df, cells, distances


def create_adata_quick(df, I, polygons):
    coordinates = (
        df.groupby(["cells"]).mean().iloc[:, [0, 1]]
    )  # calculate the mean of the transcripts for every cell. Now based on transcripts, better on masks?
    # based on masks is present in the adata.obsm

    cellCounts = (
        df.groupby(["cells", 3]).size().unstack(fill_value=0)
    )  # create a matrix based on counts
    adata = AnnData(cellCounts[cellCounts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"]

    polygonsF = polygons[
        np.isin(polygons.index.values, list(map(int, adata.obs.index.values)))
    ]
    polygonsF.index = list(map(str, polygonsF.index))
    adata.obsm["polygons"] = polygonsF

    spatial_key = "spatial"
    library_id = "melanoma"
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": I}
    adata.uns[spatial_key][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }
    return adata


def create_adata(
    df,
    cells,
    I,
    filterCrit=None,
):
    df["cellsR"] = cells
    if filterCrit is not None:
        FGenes = []
        FPositions = []
        for i, x in enumerate(distances):
            if (
                x > filterCrit
            ):  # these cells further away then this distance are filtered out
                FGenes.append(df[3][i])
                FPositions.append(i)

        df = df.drop(FPositions, axis=0)
        print("Done filtering transcripts based on distance")
    coordinates = (
        df.groupby(["cellsR"]).mean().iloc[:, [0, 1]]
    )  # calculate the mean of the transcripts for every cell. Now based on transcripts, better on masks?

    cellCounts = (
        df.groupby(["cellsR", 3]).size().unstack(fill_value=0)
    )  # create a matrix based on counts

    adata = AnnData(cellCounts)
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates

    spatial_key = "spatial"
    library_id = "melanoma"
    adata.uns[spatial_key] = {library_id: {}}
    adata.uns[spatial_key][library_id]["images"] = {}
    adata.uns[spatial_key][library_id]["images"] = {"hires": I}
    adata.uns[spatial_key][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }
    return adata


def diagnostics(adata_full, adata_filtered):
    print(
        "Of all transcripts, "
        + str(adata_filtered.X.sum() / adata_full.X.sum())
        + " is kept."
    )

    # Percentages=pd.DataFrame({'Percentage_kept':adata_filtered.X.sum(axis=0)/adata_full.X.sum(axis=0),'tot_counts':adata_filtered.X.sum(axis=0)},index=adata_full.var.index)

    adata_F = adata_full[
        adata_full.obs.index.isin(adata_filtered.obs.index),
        adata_full.var.index.isin(adata_filtered.var.index),
    ]
    adata_nuc.obs["filtered"] = adata_filtered.X.sum(axis=1) / adata_F.X.sum(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sc.pl.spatial(
        adata_nuc,
        color="filtered",
        spot_size=70,
        show=False,
        cmap="magma",
        alpha=1,
        title="Plot showing nucleus density",
        ax=ax,
    )

    return Percentages


def preprocessAdata(adata, mask):

    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[2, 5])
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.distplot(adata.obs["total_counts"], kde=False, ax=axs[0])
    sns.distplot(adata.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.raw = adata

    # nucleusSizeNormalization
    unique, counts = np.unique(mask, return_counts=True)
    nucleusSize = []
    for i in adata.obs.index:
        nucleusSize.append(counts[int(i)])
    adata.obs["nucleusSize"] = nucleusSize
    adata.X = (adata.X.T / adata.obs.nucleusSize.values).T

    # sc.pp.normalize_total(adata) #This no
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack", n_comps=50)
    sc.pl.pca(adata, color="total_counts")
    sc.pl.pca_variance_ratio(adata, n_pcs=50)  # lets take 6,10 or 12

    return adata


def preprocess3(adata, pcs, neighbors, spot_size=70):

    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.umap(adata)
    # sc.pl.umap(adata, color=['Folr2','Glul','Sox9','Cd9']) #total counts doesn't matter that much
    sc.tl.leiden(adata, resolution=0.5)
    sc.pl.umap(adata, color=["leiden"])
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sc.pl.spatial(
        adata,
        color="leiden",
        spot_size=spot_size,
        show=False,
        cmap="magma",
        alpha=1,
        title="Spatial location of clusters",
        ax=ax,
    )
    return adata


def score_genes(adata, path_marker_genes, RowNorm=False):

    df_markers = pd.read_csv(path_marker_genes, index_col=0)
    df_markers.columns = df_markers.columns.str.replace("Tot_Score_", "")
    df_markers.columns = df_markers.columns.str.replace("uppfer", "upffer")
    Dict = {}
    for i in df_markers:
        Genes = []
        for row, value in enumerate(df_markers[i]):
            if value > 0:
                Genes.append(df_markers.index[row])
                # print(df_markers.index[row])
        Dict[i] = Genes

    for i in Dict:
        sc.tl.score_genes(adata, Dict[i], score_name=i)

    # ScoresperCluster = adata.obs[[col for col in adata.obs if col.startswith('Tot')]] #very specific to this dataset
    ScoresperCluster = adata.obs[
        [col for col in adata.obs if col in (df_markers.columns)]
    ]
    if RowNorm == True:
        RowNorm = ScoresperCluster.sub(
            ScoresperCluster.mean(axis=1).values, axis="rows"
        ).div(
            ScoresperCluster.std(axis=1).values, axis="rows"
        )  # row normalization
        # Row normalization is just there for visualization purposes, to make sure we are not overdoing it

        adata.obs[ScoresperCluster.columns.values] = RowNorm
        Temp = pd.DataFrame(np.sort(RowNorm)[:, -2:])
    else:
        Temp = pd.DataFrame(np.sort(ScoresperCluster)[:, -2:])
    Scores = (Temp[1] - Temp[0]) / ((Temp[1] + Temp[0]) / 2)
    adata.obs["Cleanliness"] = Scores.values
    adata.obs["maxScores"] = ScoresperCluster.idxmax(axis=1)

    sc.pl.umap(adata, color=["Cleanliness", "maxScores"])
    sc.pl.umap(adata, color=["leiden", "maxScores"])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sc.pl.spatial(
        adata,
        color="maxScores",
        spot_size=70,
        show=False,
        cmap="magma",
        alpha=1,
        title="AnnotationScores",
        ax=ax,
    )

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sc.pl.spatial(
        adata,
        color="Cleanliness",
        spot_size=70,
        show=False,
        cmap="magma",
        alpha=1,
        title="Cleanliness of data",
        ax=ax,
    )

    sc.pl.heatmap(adata, var_names=ScoresperCluster.columns.values, groupby="leiden")

    sc.pl.heatmap(
        adata[
            adata.obs.leiden.isin(
                ["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
            )
        ],
        var_names=ScoresperCluster.columns.values,
        groupby="leiden",
    )
    return Dict


def score_gene_mel(adata, path_marker_genes, Normalize=False):

    markers = pd.read_csv(path_marker_genes, header=None, index_col=0)
    MarkerGenesDict = markers.T.to_dict("list")
    for i in MarkerGenesDict:
        MarkerGenesDict[i] = [x for x in MarkerGenesDict[i] if str(x) != "nan"]

    for i in MarkerGenesDict:
        sc.tl.score_genes(adata, MarkerGenesDict[i], score_name=i)

    ScoresperCluster = adata.obs[
        [col for col in adata.obs if col[0].isupper()]
    ]  # very specific to this dataset
    if Normalize == True:
        RowNorm = ScoresperCluster.sub(
            ScoresperCluster.mean(axis=1).values, axis="rows"
        ).div(
            ScoresperCluster.std(axis=1).values, axis="rows"
        )  # row normalization
        # Row normalization is just there for visualization purposes, to make sure we are not overdoing it

        adata.obs[ScoresperCluster.columns.values] = RowNorm
        Temp = pd.DataFrame(np.sort(RowNorm)[:, -2:])
    else:
        Temp = pd.DataFrame(np.sort(ScoresperCluster)[:, -2:])

    Scores = (Temp[1] - Temp[0]) / ((Temp[1] + Temp[0]) / 2)
    adata.obs["cleanliness"] = Scores.values
    adata.obs["maxScores"] = ScoresperCluster.idxmax(axis=1)

    sc.pl.umap(adata, color=["cleanliness", "maxScores"])
    sc.pl.umap(adata, color=["leiden", "maxScores"])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sc.pl.spatial(
        adata,
        color="maxScores",
        spot_size=50,
        show=False,
        cmap="magma",
        alpha=1,
        title="AnnotationScores",
        ax=ax,
    )

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    # sc.pl.spatial(adata,color='Cleanliness',spot_size=50,show=False,cmap='magma',alpha=1,title='Cleanliness of data',ax=ax)
    sc.pl.spatial(
        adata,
        color="Sox10",
        spot_size=50,
        show=False,
        cmap="magma",
        alpha=1,
        title="Sox10",
        ax=ax,
    )
    sc.pl.heatmap(adata, var_names=ScoresperCluster.columns.values, groupby="leiden")

    sc.pl.heatmap(
        adata[adata.obs.leiden.isin(["5", "6", "7", "8", "9", "10", "11"])],
        var_names=ScoresperCluster.columns.values,
        groupby="leiden",
    )
    return MarkerGenesDict


def clustercleanliness(adata, I, celltypes, crop_coord=[0, 2000, 0, 2000]):

    adata.obs["maxScores"] = adata.obs[
        [col for col in adata.obs if col in (celltypes)]
    ].idxmax(axis=1)

    adata.obs.maxScores = adata.obs.maxScores.astype("category")
    Other_ImmuneCells = celltypes[[1, 4, 7, 8, 9, 10, 11, 12, 13, 17]]
    vein = celltypes[[15, 18]]
    adata.obs["maxScoresSave"] = adata.obs.maxScores

    adata.obs.maxScores = adata.obs.maxScores.cat.add_categories(["Other_ImmuneCells"])
    for i in range(0, len(adata.obs.maxScores)):
        if adata.obs.maxScores[i] in Other_ImmuneCells:
            adata.obs.maxScores[i] = "Other_ImmuneCells"

    adata.obs.maxScores = adata.obs.maxScores.cat.add_categories(["vein_EC45"])
    for i in range(0, len(adata.obs.maxScores)):
        if adata.obs.maxScores[i] in vein:
            adata.obs.maxScores[i] = "vein_EC45"

    adata.obs.maxScoresSave = adata.obs.maxScoresSave.cat.add_categories(["vein_EC45"])
    for i in range(0, len(adata.obs.maxScores)):
        if adata.obs.maxScoresSave[i] in vein:
            adata.obs.maxScoresSave[i] = "vein_EC45"
    for i in Other_ImmuneCells:
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

    color_Dict = dict(
        zip(list(adata.obs.maxScores.cat.categories), adata.uns["maxScores_colors"])
    )
    for i, name in enumerate(color_Dict.keys()):
        color_Dict[name] = colors[i]

    colorsI = [
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
        color_Dict[Other_ImmuneCells[i]] = colorsI[i]

    # create the plots

    Stacked = (
        adata.obs.groupby(["leiden", "maxScores"], as_index=False)
        .size()
        .pivot("leiden", "maxScores")
        .fillna(0)
    )
    Stacked_Norm = Stacked.div(Stacked.sum(axis=1), axis=0)
    Stacked_Norm.columns = list(adata.obs.maxScores.cat.categories)

    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    Stacked_Norm.plot(
        kind="bar", stacked=True, ax=f.gca(), color=color_Dict
    )  # .legend(loc='lower left')
    # ax.axis('off')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.title("Cluster purity based on marker gene lists", fontsize="xx-large")
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(I, cmap="gray")
    sc.pl.spatial(
        adata,
        color="maxScores",
        spot_size=np.sqrt(adata.obs.nucleusSize),
        show=False,
        cmap="magma",
        ax=ax,
        scale_factor=1,
        img_key=None,
        alpha_img=0,
        alpha=1,
    )
    plt.title("Spatial organization of assigned celltypes", fontsize="xx-large")
    plt.gca().invert_yaxis()
    ax.axis("off")

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(I, cmap="gray")
    sc.pl.spatial(
        adata,
        color="maxScores",
        spot_size=np.sqrt(adata.obs.nucleusSize),
        show=False,
        cmap="magma",
        ax=ax,
        scale_factor=1,
        img_key=None,
        alpha_img=0,
        alpha=1,
        palette=color_Dict,
        crop_coord=crop_coord,
    )
    plt.title(
        "Zoom in on Spatial organization of assigned celltypes", fontsize="xx-large"
    )
    plt.gca().invert_yaxis()
    ax.axis("off")

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(adata, color=["maxScores"], ax=ax, size=60, show=False)
    ax.axis("off")
    plt.title("UMAP colored by annotation based on celltype ", fontsize="xx-large")
    plt.show()
