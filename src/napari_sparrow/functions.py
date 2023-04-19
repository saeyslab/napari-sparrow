""" This file holds all the general functions that are used to build up the pipeline and the notebooks. The functions are odered by their occurence in the pipeline from top to bottom."""

# %load_ext autoreload
# %autoreload 2
import warnings
from itertools import chain
from typing import List, Optional, Tuple

import cv2
import geopandas
import matplotlib
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scanpy as sc
import seaborn as sns
import shapely
import squidpy as sq
import torch
from anndata import AnnData
from basicpy import BaSiC
from cellpose import models
from rasterio import features
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import dask.dataframe as dd

import xarray as xr
import dask
import numpy as np
import hvplot.xarray
import numpy as np
from longsgis import voronoiDiagram4plg
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from scipy.ndimage.filters import gaussian_filter



def read_in_zarr(path_to_zarr_file,zyx_order=[0,1,2],subset=None):
    """This function read in a zarr file containing the tissue image. If a z-stack is present, automatically, a z-projection is performed.
     A quidpy imagecontainer is returned."""
    da = dask.array.from_zarr(path_to_zarr_file)
    da = xr.DataArray(
    da, 
    dims=('z', 'y', 'x'),
    coords={
        "z": np.arange(da.shape[zyx_order[0]]),
        "y": np.arange(da.shape[zyx_order[1]]), 
        "x": np.arange(da.shape[zyx_order[2]]),
    }
    )
    if da.z.shape[0]>1:
        da=da.max(dim='z')
    if subset:    
        da=da[subset[0]:subset[1],subset[2]:subset[3]]    
    ic = sq.im.ImageContainer(da)    

    return ic

def create_subset_image(ic,crd):
    """Reads in sq image container and returns a subset in a sq.imagecontainer"""
    Xmax=ic.data.sizes['x']
    Ymax=ic.data.sizes['y']    
    img=ic.data.assign_coords({'x':np.arange(Xmax),'y':np.arange(Ymax)})  
    I_small=img['image'].sel(x=slice(crd[0],crd[1]),y=slice(crd[2],crd[3]))
    ic_test= sq.im.ImageContainer(I_small)
    return ic_test


def tilingCorrection(
    img: sq.im.ImageContainer,
    left_corner: Tuple[int, int] = None,
    size: Tuple[int, int] = None,
    tile_size: int = 2144,
) -> Tuple[sq.im.ImageContainer, np.ndarray]:
    """Returns the corrected image and the flatfield array

    This function corrects for the tiling effect that occurs in some image data for example the resolve dataset.
    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.
    """

    # Create the tiles
    tiles = img.generate_equal_crops(size=tile_size, as_array="image")
    tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])
    black=np.array([1 if ~np.any(tile-1) else 0 for tile in tiles]) # determine if 

    #create the masks for inpainting 
    i_mask = (np.block(
        [
            list(tiles[i : i + (img.shape[1] // tile_size)])
            for i in range(0, len(tiles), img.shape[1] // tile_size)
        ]
    ).astype(np.uint16)==0)
    if tiles.shape[0]<5:
        print('There aren\'t enough tiles to perform tiling correction (less than 5). This step will be skipped.')
        tiles_corrected=tiles
        flatfield=None
    else:
        basic = BaSiC(smoothness_flatfield=1)
        basic.fit(tiles)
        flatfield = basic.flatfield
        tiles_corrected = basic.transform(tiles)


    tiles_corrected = np.array(
        [tiles[number] if black[number]==1 else tile for number,tile in enumerate(tiles_corrected)]
    )

    # Stitch the tiles back together
    i_new = np.block(
        [
            list(tiles_corrected[i : i + (img.shape[1] // tile_size)])
            for i in range(0, len(tiles_corrected), img.shape[1] // tile_size)
        ]
    ).astype(np.uint16)

    

    img = sq.im.ImageContainer(i_new, layer="image")
    img.add_img(
        i_mask.astype(np.uint8),
        layer="mask",
    )

    if size is not None and left_corner is not None:
        img = img.crop_corner(*left_corner, size)

    # Perform inpainting
    img = img.apply(
        {"0": cv2.inpaint},
        layer="image",
        drop=True,
        channel=0,
        copy=True,
        fn_kwargs={
            "inpaintMask": img.data.mask.squeeze().to_numpy(),
            "inpaintRadius": 55,
            "flags": cv2.INPAINT_NS,
        },
    )

    return img, flatfield


def tilingCorrectionPlot(
    img: np.ndarray, flatfield, img_orig: np.ndarray, output: str = None
) -> None:
    """Creates the plots based on the correction overlay and the original and corrected images."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Tile correction overlay
    if flatfield is not None:
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
        ax1.imshow(flatfield, cmap="gray")
        ax1.set_title("Correction performed per tile")

        # Save the plot to ouput
        if output:
            plt.close(fig1)
            fig1.savefig(output + "0.png")

    # Original and corrected image
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
    ax2[0].imshow(img, cmap="gray")
    ax2[0].set_title("Corrected image")
    ax2[1].imshow(img_orig, cmap="gray")
    ax2[1].set_title("Original image")
    ax2[0].spines["top"].set_visible(False)
    ax2[0].spines["right"].set_visible(False)
    ax2[0].spines["bottom"].set_visible(False)
    ax2[0].spines["left"].set_visible(False)
    ax2[1].spines["top"].set_visible(False)
    ax2[1].spines["right"].set_visible(False)
    ax2[1].spines["bottom"].set_visible(False)
    ax2[1].spines["left"].set_visible(False)
    
    # Save the plot to ouput
    if output:
        plt.close(fig2)
        fig2.savefig(output + "1.png")


def preprocessImage(
    img: sq.im.ImageContainer,
    contrast_clip: float = 2.5,
    size_tophat: int = None,
) -> sq.im.ImageContainer:
    """Returns the new image

    This function performs the preprocessing of the image.
    Contrast_clip indicates the input to the create_CLAHE function for histogram equalization.
    Size_tophat indicates the tophat filter size. If no tophat lfiter size is given, no tophat filter is applied. The recommendable size is 45.
    Small_size_vis indicates the coordinates of an optional zoom in plot to check the processing better.
    """

    # Apply tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(
            img.data.image.squeeze().to_numpy(), size_tophat
        )
        max_of_min_t = ndimage.maximum_filter(minimum_t, size_tophat)
        img = img.apply(
            {"0": lambda array: array - max_of_min_t},
            layer="image",
            drop=True,
            channel=0,
            copy=True,
        )

    # Enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    img = img.apply({"0": clahe.apply}, layer="image", drop=True, channel=0, copy=True)

    return img


def preprocessImagePlot(
    img: np.ndarray,
    img_orig: np.ndarray,
    small_size_vis: List[int] = None,
    output: str = None,
) -> None:
    """Creates the plots based on the original and preprocessed image."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Original and preprocessed image
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
    ax1[0].imshow(img, cmap="gray")
    ax1[0].set_title("Corrected image")
    ax1[1].imshow(img_orig, cmap="gray")
    ax1[1].set_title("Original image")

    # Save the plot to ouput
    if output:
        plt.close(fig1)
        fig1.savefig(output + "0.png")

    # Plot small part of the images
    if small_size_vis is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
        ax2[0].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax2[0].set_title("Corrected image")
        ax2[1].imshow(
            img_orig[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax2[1].set_title("Original image")

        # Save the plot to ouput
        if output:
            plt.close(fig2)
            fig2.savefig(output + "1.png")

def cellpose(img, min_size=80,cellprob_threshold=-4, flow_threshold=0.85, diameter=100, model_type="cyto",channels=[1,0],device='cpu'):
    gpu=torch.cuda.is_available()
    model = models.Cellpose(gpu=gpu,model_type=model_type,device=torch.device(device))
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks   
            
def segmentation(
    img: sq.im.ImageContainer,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels =[0, 0],
    ): 

    
    sq.im.segment(img=img, layer="image", method=cellpose,chunks='auto',min_size=min_size,cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold, diameter=diameter, model_type=model_type,channels=channels,device=device)
    masks=img.data.segmented_custom.squeeze().to_numpy()
    mask_i = np.ma.masked_where(masks == 0, masks)

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    #polygons["border_color"] = polygons.geometry.map(fc.border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    #polygons["color"] = polygons.geometry.map(fc.color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")
    return masks, mask_i, polygons, img            
            

def segmentationDeprecated(
    img: sq.im.ImageContainer,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
) -> Tuple[np.ndarray, np.ndarray, geopandas.GeoDataFrame, sq.im.ImageContainer]:
    """Returns the segmentation masks, the image masks and the polygons

    This function segments the data, using the cellpose algorithm.
    Img is the input image.
    You can define your device by setting the device parameter.
    Min_size indicates the minimal amount of pixels in a mask.
    The flow_threshold indicates someting about the shape of the masks, if you increase it, more masks with less orund shapes will be accepted.
    The diameter is a very important parameter to estimate, in the best case you estimate it yourself. It indicates the mean expected diameter of your dataset.
    If you put None in diameter, the model will estimate it automatically.
    Mask_threshold indicates how many of the possible masks are kept. Making it smaller (up to -6), will give you more masks, bigger is less masks.
    When an RGB image is given an input, the R channel is expected to have the nuclei, and the blue channel the membranes.
    When whole cell segmentation needs to be performed, model_type=cyto, otherwise, model_type=nuclei.
    """

    channels = np.array(channels)

    # Perform cellpose segmentation
    model = models.Cellpose(device=torch.device(device), model_type=model_type)
    masks, _, _, _ = model.eval(
        img.data.image.squeeze().to_numpy(),
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    mask_i = np.ma.masked_where(masks == 0, masks)

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    #polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    #polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")
    polygons.index = list(map(str, polygons.index))

    img.add_img(masks, layer="segment_cellpose")

    return masks, mask_i, polygons, img


def segmentationPlot(
    ic,
    mask_i=None,
    polygons: geopandas.GeoDataFrame=None,
    crd=None,
    channels=[0,0],
    small_size_vis=None,
    img_layer='image',
    output: str = None,
) -> None:
    if output:
        plt.ioff()
    if small_size_vis:
        crd=small_size_vis
    if polygons is None:
        raise ValueError("No polygons are given as input") 
    if type(ic)==np.ndarray:
        ic = sq.im.ImageContainer(ic)
        
    Xmax=ic.data.sizes['x']
    Ymax=ic.data.sizes['y']    
    ic=ic.data.assign_coords({'x':np.arange(Xmax),'y':np.arange(Ymax)})  
    #crd=small_size_vis
    #crd=[2000,3500,1000,2500]
    if crd==None:
        crd=[0,Ymax,0,Xmax]
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    ic[img_layer].squeeze().sel(x=slice(crd[2],crd[3]),y=slice(crd[0],crd[1])).plot.imshow(cmap='gray', robust=True, ax=ax[0],add_colorbar=False)
    ic[img_layer].squeeze().sel(x=slice(crd[2],crd[3]),y=slice(crd[0],crd[1])).plot.imshow(cmap='gray', robust=True, ax=ax[1],add_colorbar=False)
    polygons.cx[crd[2]:crd[3],crd[0]:crd[1]].plot(
            ax=ax[1],
            edgecolor="white",
            linewidth=1,
            alpha=0.5,
            legend=True,
            aspect=1,
            )
    for i in range(len(ax)):
        ax[i].axes.set_aspect('equal')
        ax[i].set_xlim(crd[2], crd[3])
        ax[i].set_ylim(crd[0], crd[1])
        ax[i].invert_yaxis()
        ax[i].set_title("")
        #ax.axes.xaxis.set_visible(False)
        #ax.axes.yaxis.set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
    
    # Save the plot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + ".png")            
        

def mask_to_polygons_layer(mask: np.ndarray) -> geopandas.GeoDataFrame:
    """Returns the polygons as GeoDataFrame

    This function converts the mask to polygons.
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """

    all_polygons = []
    all_values = []

    # Extract the polygons from the mask
    for shape, value in features.shapes(
        mask.astype(np.int32),
        mask=(mask > 0),
        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shapely.geometry.shape(shape))
        all_values.append(int(value))

    return geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)


def color(_) -> matplotlib.colors.Colormap:
    """Select random color from set1 colors."""
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    """Select border color from tab10 colors or preset color (1, 1, 1, 1) otherwise."""
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    """Select linewidth 1 if true else 0.5."""
    return 1 if r else 0.5

def delete_overlap(voronoi,polygons):    
    I1,I2=voronoi.sindex.query_bulk(voronoi['geometry'], predicate="overlaps")  
    voronoi2=voronoi.copy()

    for cell1, cell2 in zip(I1,I2):

        #if cell1!=cell2:
        voronoi.geometry.iloc[cell1]=voronoi.iloc[cell1].geometry.intersection(voronoi2.iloc[cell1].geometry.difference(voronoi2.iloc[cell2].geometry))
        voronoi.geometry.iloc[cell2]=voronoi.iloc[cell2].geometry.intersection(voronoi2.iloc[cell2].geometry.difference(voronoi2.iloc[cell1].geometry))
    voronoi['geometry']=voronoi.geometry.union(polygons.geometry) 
    return voronoi

def allocation(ddf,ic: sq.im.ImageContainer, masks: np.ndarray=None, library_id: str = "spatial_transcriptomics",radius=0,polygons=None, verbose=False
    ):
    
    # Create the polygon shapes for the mask
    if polygons is None:
        polygons = mask_to_polygons_layer(masks)
        polygons["cells"] = polygons.index
        polygons = polygons.dissolve(by="cells")
    polygons.index = list(map(str, polygons.index))
    
    #calculate new mask based on radius    
    if radius!=0:
        boundary=Polygon([(0,0),(ic.shape[1]+200,0),(ic.shape[1]+200,ic.shape[0]+200),(0,ic.shape[0]+200)])
        polygons['geometry']=polygons.simplify(2)
        vd = voronoiDiagram4plg(polygons, boundary)
        voronoi=geopandas.sjoin(vd,polygons,predicate='contains',how='left')
        voronoi.index=voronoi.index_right
        voronoi=voronoi[~voronoi.index.duplicated(keep='first')]
        voronoi=delete_overlap(voronoi,polygons)
        buffered=polygons.buffer(distance=radius)
        intersected=voronoi.sort_index().intersection(buffered.sort_index())
        polygons.geometry=intersected
        
        masks=rasterio.features.rasterize( #it should be possible to give every shape  number. You need to give the value with it as input. 
            zip(intersected.geometry,intersected.index.values.astype(float)), 
            out_shape=[ic.shape[0],ic.shape[1]],dtype='uint32')    
        
    # adapt transcripts file     
    ddf = ddf[
    (ic.data.attrs["coords"].y0 < ddf['y'])
    & (ddf['y'] < masks.shape[0] + ic.data.attrs["coords"].y0)
    & (ic.data.attrs["coords"].x0 < ddf['x'])
    & (ddf['x'] < masks.shape[1] + ic.data.attrs["coords"].x0)
    ]
    if verbose:
        print('Started df calculation')

    df=ddf.compute()
    if verbose:
        print('df calculated')
    df["cells"] = masks[
        df['y'].values.astype(int) - ic.data.attrs["coords"].y0,
        df['x'].values.astype(int) - ic.data.attrs["coords"].x0,
    ]
    if masks is None:
        masks=ic.data.segment_cellpose.squeeze().to_numpy()
    # Calculate the mean of the transcripts for every cell
    coordinates = df.groupby(["cells"]).mean().loc[:, ['x', 'y']]
    cell_counts = df.groupby(["cells", "gene"]).size().unstack(fill_value=0)
    if verbose: 
        
        print('finished groupby')
        # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0],dtype='int64')
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values
    if verbose:
        print('created anndata object')
    # Add the polygons to the anndata object
    polygons["linewidth"] = polygons.geometry.map(linewidth)

    polygons_f = polygons[
        np.isin(polygons.index.values, adata.obs.index.values)
    ]
    #polygons_f.index = list(map(str, polygons_f.index))
    adata.obsm["polygons"] = polygons_f
    adata.obs["cell_ID"] = [int(x) for x in adata.obsm["polygons"].index]

    # Add the figure to the anndata object
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {
        "hires": ic.data.image.squeeze().to_numpy()
        #"segmentation":ic.data.segment_cellpose.squeeze().to_numpy()
    }
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }
    #adata.uns["spatial"][library_id]["segmentation"] = masks.astype(np.uint16)

    return adata,df

def control_transcripts(df,scaling_factor=100):
    """This function plots the transcript density of the tissue. You can use it to compare different regions in your tissue on transcript density.  """
    Try=df.groupby(['x','y']).count()['gene']
    Image=np.array(Try.unstack(fill_value=0))
    Image=Image/np.max(Image)
    blurred = gaussian_filter(scaling_factor*Image, sigma=7)
    return blurred

def plot_control_transcripts(blurred,img,crd=None):
    if crd:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))

        ax[0].imshow(blurred.T[crd[0]:crd[1],crd[2]:crd[3]],cmap='magma',vmax=5)
        ax[1].imshow(img[crd[0]:crd[1],crd[2]:crd[3]],cmap='gray')
        ax[0].set_title("Transcript density")
        ax[1].set_title("Corrected image")
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(blurred.T,cmap='magma',vmax=5)
    ax[1].imshow(img,cmap='gray')
    ax[0].set_title("Transcript density")
    ax[1].set_title("Corrected image")

def analyse_genes_left_out(adata,df):
    """ This function """
    filtered=pd.DataFrame(adata.X.sum(axis=0)/df.groupby('gene').count()['x'][adata.var.index])
    filtered=filtered.rename(columns={'x':'proportion_kept'})
    filtered['raw_counts']=df.groupby('gene').count()['x'][adata.var.index]
    sns.scatterplot(data=filtered, y="proportion_kept", x="raw_counts")
    
    plt.axvline(filtered['raw_counts'].median(), color='green', linestyle='dashed')
    plt.axhline(filtered['proportion_kept'].median(), color='red', linestyle='dashed')
    plt.xlim(left=-10,right=filtered['raw_counts'].quantile(0.90)) # set y-axis limit from 0 to the 95th percentile of y
    # show the plot
    plt.show()
    print('The ten genes with the highest proportion of transcripts filtered out')
    print(filtered.sort_values(by='proportion_kept')[0:10])
    return filtered


def create_adata_quick(
    path: str, ic: sq.im.ImageContainer, masks: np.ndarray, library_id: str = "spatial_transcriptomics"
) -> AnnData:
    """Returns the AnnData object with transcript and polygon data.

    This function creates the polygon shapes from the mask and adjusts the colors and linewidth.
    The transcripts are read from the csv file in path, all transcripts within cells are assigned.
    Only cells with transcripts are retained.
    """

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    polygons["geometry"] = polygons["geometry"].translate(
        float(ic.data.attrs["coords"].x0), float(ic.data.attrs["coords"].y0)
    )

    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")

    # Allocate the transcripts
    in_df = pd.read_csv(path, delimiter="\t", header=None)
    # Changed for subset
    df = in_df[
        (ic.data.attrs["coords"].y0 < in_df[1])
        & (in_df[1] < masks.shape[0] + ic.data.attrs["coords"].y0)
        & (ic.data.attrs["coords"].x0 < in_df[0])
        & (in_df[0] < masks.shape[1] + ic.data.attrs["coords"].x0)
    ]

    df["cells"] = masks[
        df[1].values - ic.data.attrs["coords"].y0,
        df[0].values - ic.data.attrs["coords"].x0,
    ]

    # Calculate the mean of the transcripts for every cell
    coordinates = df.groupby(["cells"]).mean().iloc[:, [0, 1]]
    cell_counts = df.groupby(["cells", 3]).size().unstack(fill_value=0)

    # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"]#.values

    # Add the polygons to the anndata object
    polygons_f = polygons[
        np.isin(polygons.index.values, list(map(int, adata.obs.index.values)))
    ]
    polygons_f.index = list(map(str, polygons_f.index))
    adata.obsm["polygons"] = polygons_f
    adata.obs["cell_ID"] = [int(x) for x in adata.obsm["polygons"].index]

    # Add the figure to the anndata object
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {
        "hires": ic.data.image.squeeze().to_numpy()
    }
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }
    adata.uns["spatial"][library_id]["segmentation"] = masks.astype(np.uint16)
    adata.uns["spatial"][library_id]["points"] = AnnData(in_df.values[:, 0:2])
    adata.uns["spatial"][library_id]["points"].obs = pd.DataFrame(
        {"gene": in_df.values[:, 3]}
    )

    return adata

def plot_shapes(
    adata,
    column: str = None,
    cmap: str = "magma",
    img=None,
    img_layer='image',
    alpha: float = 0.5,
    library_id='spatial_transcriptomics',
    crd=None,
    output: str = None,
    vmin=None,
    vmax=None,
) -> None:
    
    adata.obsm['polygons']=geopandas.GeoDataFrame(adata.obsm['polygons'],geometry=adata.obsm['polygons']['geometry'])
    
    if img==None:
        img= sq.im.ImageContainer(adata.uns["spatial"][library_id]["images"]["hires"], layer="image")
        
    Xmax=img.data.sizes['x']
    Ymax=img.data.sizes['y']    
    img=img.data.assign_coords({'x':np.arange(Xmax),'y':np.arange(Ymax)})  
    
    if crd==None:
        crd=[0,Xmax,0,Ymax]
        
    if column is not None:
        if column + "_colors" in adata.uns:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "new_map",
                adata.uns[column + "_colors"],
                N=len(adata.uns[column + "_colors"]),
            ) 
        if column in adata.obs.columns:    
            column=adata[adata.obsm["polygons"].cx[crd[0]:crd[1],crd[2]:crd[3]].index,:].obs[column]
        elif column in adata.var.index:
            column=adata[adata.obsm["polygons"].cx[crd[0]:crd[1],crd[2]:crd[3]].index,:].X[:,np.where(adata.var.index==column)[0][0]]
        else: 
            print('The column defined in the function isnt a column in obs, nor is it a gene name, the plot is made without taking into account this value.')
            column= None
            cmap=None
    else:
        cmap=None
    if vmin!=None:
        vmin=np.percentile(column,vmin)
    if vmax!=None:
        vmax=np.percentile(column,vmax)    
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    img[img_layer].squeeze().sel(x=slice(crd[0],crd[1]),y=slice(crd[2],crd[3])).plot.imshow(cmap='gray', robust=True, ax=ax,add_colorbar=False)

    adata.obsm["polygons"].cx[crd[0]:crd[1],crd[2]:crd[3]].plot(
            ax=ax,
            edgecolor="white",
            column=column,
            linewidth=1,
            alpha=alpha,
            legend=True,
            aspect=1,
            cmap=cmap,
            vmax=vmax,#np.percentile(column,vmax),
            vmin=vmin,#np.percentile(column,vmin)
            )
    
    ax.axes.set_aspect('equal')
    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])
    ax.invert_yaxis()
    ax.set_title("")
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Save the plot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + ".png")
    else:
        plt.show()

def preprocessAdata(
    adata: AnnData, mask: np.ndarray, nuc_size_norm: bool = True, n_comps: int = 50
) -> Tuple[AnnData, AnnData]:
    """Returns the new and original AnnData objects

    This function calculates the QC metrics.
    All cells with les then 10 genes and all genes with less then 5 cells are removed.
    Normalization is performed based on the size of the nucleus in nuc_size_norm.
    """

    # Calculate QC Metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[2, 5])
    adata_orig = adata

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.raw= adata

    # Normalize nucleus size
    
    indices, counts = np.unique(mask, return_counts=True)
    adata.obs["nucleusSize"] = [counts[np.where(indices==int(index))][0] for index in adata.obs.index]
    if nuc_size_norm:   
        adata.X = (adata.X.T*100 / adata.obs.nucleusSize.values).T

        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps)
    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )

    return adata, adata_orig


def preprocesAdataPlot(adata: AnnData, adata_orig: AnnData, output: str = None) -> None:
    """This function plots the size of the nucleus related to the counts."""

    # disable interactive mode
    if output:
        plt.ioff()

    sc.pl.pca(adata, color="total_counts", show=not output)

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.histplot(adata_orig.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(adata_orig.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])
    plt.show()
    plt.scatter(adata.obs["nucleusSize"], adata.obs["total_counts"])
    plt.title = "cellsize vs cellcount"

    # Save the plot to ouput
    if output:
        plt.close()
        plt.savefig(output + "0.png")
        plt.close(fig)
        fig.savefig(output + "1.png")


def filter_on_size(
    adata: AnnData, min_size: int = 100, max_size: int = 100000
) -> Tuple[AnnData, int]:
    """Returns a tuple with the AnnData object and the number of filtered cells.

    All cells outside of the min and max size range are removed.
    If the distance between the location of the transcript and the center of the polygon is large, the cell is deleted.
    """

    start = adata.shape[0]

    # Calculate center of the cell and distance between transcript and polygon center
    adata.obsm["polygons"]["X"] = adata.obsm["polygons"].centroid.x
    adata.obsm["polygons"]["Y"] = adata.obsm["polygons"].centroid.y
    adata.obs["distance"] = np.sqrt(
        np.square(adata.obsm["polygons"]["X"] - adata.obsm["spatial"][:, 0])
        + np.square(adata.obsm["polygons"]["Y"] - adata.obsm["spatial"][:, 1])
    )

    # Filter cells based on size and distance
    #adata = adata[adata.obs["nucleusSize"] < max_size, :]
    #adata = adata[adata.obs["nucleusSize"] > min_size, :]
    adata=adata[adata.obsm['polygons'].area>min_size,:]
    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )
    adata=adata[adata.obsm['polygons'].area<max_size,:]

    adata = adata[adata.obs["distance"] < 70, :]
    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )
    filtered = start - adata.shape[0]

    return adata, filtered


def extract(ic: sq.im.ImageContainer, adata: AnnData) -> AnnData:
    """This function performs segmenation feature extraction and adds cell area and mean intensity to the annData object under obsm segmentation_features."""
    sq.im.calculate_image_features(
        adata,
        ic,
        layer="image",
        features="segmentation",
        key_added="segmentation_features",
        features_kwargs={
            "segmentation": {
                "label_layer": "segment_cellpose",
                "props": ["label", "area", "mean_intensity"],
                "channels": [0],
            }
        },
    )

    return adata


def clustering(
    adata: AnnData, pcs: int, neighbors: int, cluster_resolution: float = 0.8
) -> AnnData:
    """Returns the AnnData object.

    Performs neighborhood analysis, Leiden clustering and UMAP.
    Provides option to save the plots to output.
    """

    # Neighborhood analysis
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.umap(adata)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=cluster_resolution)
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

    return adata


def clustering_plot(adata: AnnData, output: str = None) -> None:
    """This function plots the clusters and genes ranking"""

    # disable interactive mode
    if output:
        plt.ioff()

    # Leiden clustering
    sc.pl.umap(adata, color=["leiden"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "0.png", bbox_inches="tight")
        plt.close()
        sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False, show=False)
        plt.savefig(output + "1.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False)


def scoreGenes(
    adata: AnnData,
    path_marker_genes: str,
    row_norm: bool = False,
    repl_columns: dict[str, str] = None,
    del_genes: List[str] = None,
    input_dict=False
) -> Tuple[dict, pd.DataFrame]:
    """Returns genes dict and the scores per cluster

    Load the marker genes from csv file in path_marker_genes.
    repl_columns holds the column names that should be replaced the in the marker genes.
    del_genes holds the marker genes that should be deleted from the marker genes and genes dict.
    """

    # Load marker genes from csv
    if input_dict:
        df_markers=pd.read_csv(path_marker_genes,header=None,index_col=0)
        df_markers=df_markers.T
        genes_dict=df_markers.to_dict('list')
        for i in genes_dict:
            genes_dict[i]=[x for x in genes_dict[i] if str(x)!='nan']
    # Replace column names in marker genes
    else:
        df_markers = pd.read_csv(path_marker_genes, index_col=0)
        if repl_columns:
            for column, replace in repl_columns.items():
                df_markers.columns = df_markers.columns.str.replace(column, replace)

        # Create genes dict with all marker genes for every celltype
        genes_dict = {}
        for i in df_markers:
            genes = []
            for row, value in enumerate(df_markers[i]):
                if value > 0:
                    genes.append(df_markers.index[row])
            genes_dict[i] = genes

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(adata, value, score_name=key)
        except ValueError:
            warnings.warn(
                f"Markergenes {value} not present in region, celltype {key} not found"
            )

    # Delete genes from marker genes and genes dict
    if del_genes:
        for gene in del_genes:
            del df_markers[gene]
            del genes_dict[gene]

    scoresper_cluster = adata.obs[
        [col for col in adata.obs if col in df_markers.columns]
    ]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(
            scoresper_cluster.mean(axis=1).values, axis="rows"
        ).div(scoresper_cluster.std(axis=1).values, axis="rows")
        adata.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    adata.obs["Cleanliness"] = scores.values
    adata.obs["annotation"] = scoresper_cluster.idxmax(axis=1)
    adata.obs["annotation"] = adata.obs["annotation"].astype('category')
    return genes_dict, scoresper_cluster


def scoreGenesPlot(
    adata: AnnData,
    scoresper_cluster: pd.DataFrame,
    filter_index: int = 5,
    output: str = None,
    library_id='spatial_transcriptomics'
) -> None:
    """This function plots the cleanliness and the leiden score next to the annotation."""

    # Custom colormap:
    colors = np.concatenate(
        (plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20)))
    )
    colors = [mpl.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # disable interactive mode
    if output:
        plt.ioff()

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(adata, color=["Cleanliness", "annotation"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "0.png", bbox_inches="tight")
        plt.close()
        sc.pl.umap(adata, color=["leiden", "annotation"], show=False)
        plt.savefig(output + "1.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.umap(adata, color=["leiden", "annotation"])

    # Plot annotation and cleanliness columns of AnnData object
    adata.uns["annotation_colors"] = colors
    plot_shapes(adata, column="annotation", output=output + "2" if output else None,library_id=library_id)
    plot_shapes(adata, column="Cleanliness", output=output + "3" if output else None,library_id=library_id)

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        adata,
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
        show=not output,
    )

    # Save the plot to ouput
    if output:
        plt.savefig(output + "4.png", bbox_inches="tight")
        plt.close()
        sc.pl.heatmap(
            adata[
                adata.obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(adata.obs.leiden))]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
            show=False,
        )
        plt.savefig(output + "5.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.heatmap(
            adata[
                adata.obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(adata.obs.leiden))]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
        )


def correct_marker_genes(
    adata: AnnData, genes: dict[str, Tuple[float, float]]
) -> AnnData:
    """Returns the new AnnData object.

    Corrects marker genes that are higher expessed by dividing them.
    The genes has as keys the genes that should be corrected and as values the threshold and the divider.
    """

    # Correct for all the genes
    for gene, values in genes.items():
        for i in range(0, len(adata.obs)):
            if adata.obs[gene].iloc[i] < values[0]:
                adata.obs[gene].iloc[i] = adata.obs[gene].iloc[i] / values[1]
    return adata


def annotate_maxscore(types: str, indexes: dict, adata: AnnData) -> AnnData:
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    adata.obs.annotation = adata.obs.annotation.cat.add_categories([types])
    for i, val in enumerate(adata.obs.annotation):
        if val in indexes[types]:
            adata.obs.annotation[i] = types
    return adata


def remove_celltypes(types: str, indexes: dict, adata: AnnData) -> AnnData:
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in adata.obs.annotation.cat.categories:
            adata.obs.annotation = adata.obs.annotation.cat.remove_categories(index)
    return adata


def clustercleanliness(
    adata: AnnData,
    genes: List[str],
    gene_indexes: dict[str, int] = None,
    colors: List[str] = None,
) -> Tuple[AnnData, Optional[dict]]:
    """Returns a tuple with the AnnData object and the color dict."""
    celltypes = np.array(sorted(genes), dtype=str)
    color_dict = None

    adata.obs["annotation"] = adata.obs[
        [col for col in adata.obs if col in celltypes]
    ].idxmax(axis=1)
    adata.obs.annotation = adata.obs.annotation.astype("category")

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [mpl.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)]

    adata.uns["annotation_colors"] = colors

    if gene_indexes:
        adata.obs["annotationSave"] = adata.obs.annotation
        gene_celltypes = {}

        for key, value in gene_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, indexes in gene_indexes.items():
            adata = annotate_maxscore(gene, gene_celltypes, adata)

        for gene, indexes in gene_indexes.items():
            adata = remove_celltypes(gene, gene_celltypes, adata)

        celltypes_f = np.delete(celltypes, list(chain(*gene_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(gene_indexes.keys()))
        color_dict = dict(zip(celltypes_f, adata.uns["annotation_colors"]))

    else:
        color_dict = dict(zip(celltypes, adata.uns["annotation_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    adata.uns["annotation_colors"] = list(
        map(color_dict.get, adata.obs.annotation.cat.categories.values)
    )

    return adata, color_dict


def clustercleanlinessPlot(
    adata: AnnData,
    crop_coord: List[int] = [0, 2000, 0, 2000],
    color_dict: dict = None,
    output: str = None,
    library_id='spatial_transcriptomics'
) -> None:
    """This function plots the clustercleanliness as barplots, the images with colored celltypes and the clusters."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Create the barplot
    stacked = (
        adata.obs.groupby(["leiden", "annotation"], as_index=False)
        .size()
        .pivot("leiden", "annotation")
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(adata.obs.annotation.cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Use custom colormap
    if color_dict:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca(), color=color_dict)
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")

    # Save the barplot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + "0.png", bbox_inches="tight")
    else:
        plt.show()

    # Plot images with colored celltypes
    plot_shapes(
        adata, column="annotation", alpha=0.8, output=output + "1" if output else None, library_id=library_id
    )
    plot_shapes(
        adata,
        column="annotation",
        crd=crop_coord,
        alpha=0.8,
        output=output + "2" if output else None,
        library_id=library_id
    )

    # Plot clusters
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(adata, color=["annotation"], ax=ax, show=not output)
    ax.axis("off")

    # Save the plot to ouput
    if output:
        fig.savefig(output + "3.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def enrichment(adata: AnnData) -> AnnData:
    """Returns the AnnData object.

    Performs some adaptations to save the data.
    Calculate the nhood enrichment"
    """

    # Adaptations for saving
    adata.raw.var.index.names = ["genes"]
    adata.var.index.names = ["genes"]
    # TODO: not used since napari spatialdata
    # adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="annotation")
    return adata


def enrichment_plot(adata: AnnData, output: str = None) -> None:
    """This function plots the nhood enrichment between different celltypes."""

    # disable interactive mode
    if output:
        plt.ioff()
    # remove 'nan' values from "adata.uns['annotation_nhood_enrichment']['zscore']"
    tmp = adata.uns['annotation_nhood_enrichment']['zscore']
    adata.uns['annotation_nhood_enrichment']['zscore'] = np.nan_to_num(tmp)
    sq.pl.nhood_enrichment(adata, cluster_key="annotation", method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output + ".png", bbox_inches="tight")


def save_data(adata: AnnData, output_geojson: str, output_h5ad: str):
    """Saves the ploygons to output_geojson as GeoJson object and the rest of the AnnData object to output_h5ad as h5ad file."""

    # Save polygons to geojson
    if color in adata.obsm["polygons"]:
        del adata.obsm["polygons"]["color"]
    adata.obsm["polygons"]["geometry"].to_file(output_geojson, driver="GeoJSON")
    adata.obsm["polygons"] = pd.DataFrame(
        {
            "linewidth": adata.obsm["polygons"]["linewidth"],
             #"X": adata.obsm["polygons"]["X"],
            #"Y": adata.obsm["polygons"]["Y"],
        }
    )

    # Write AnnData object to h5ad file
    adata.write(output_h5ad)

def micron_to_pixels(df, offset_x=45_000,offset_y=45_000,pixelSize=None):
    if pixelSize:
        df[x] /= pixelSize
        df[y] /= pixelSize
    if offset_x:
            df['x'] -= offset_x
    if offset_y:        
            df['y'] -= offset_y

    return df
    
def read_in_stereoSeq(path_genes,xcol='x',ycol='y',genecol='geneID',countcol='MIDCount',skiprows=0,offset=None):
    """This function read in Stereoseq input data to a dask datafrmae with predefined column names. 
    As we are working with BGI data, a column with counts is added."""
    in_df=dd.read_csv(path_genes,delimiter='\t',skiprows=skiprows)
    in_df=in_df.rename(columns={xcol:'x',ycol:'y',genecol:'gene',countcol:'counts'})
    if offset:
        in_df=micron_to_pixels(in_df,offset_x=offset[0],offset_y=offset[1])
    in_df=in_df.loc[:,['x','y','gene','counts']]
    
    in_df=in_df.dropna()
    return in_df

def read_in_RESOLVE(path_coordinates,xcol=0,ycol=1,genecol=3,filterGenes=None,offset=None):
    
    """The output of this function gives all locations of interesting transcripts in pixel coordinates matching the input image. Dask Dataframe contains columns x,y, and gene"""
    
    
    in_df = dd.read_csv(path_coordinates, delimiter="\t", header=None)
    in_df=in_df.rename(columns={xcol:'x',ycol:'y',genecol:'gene'})
    if offset:
        in_df=micron_to_pixels(in_df,offset_x=offset[0],offset_y=offset[1])
    in_df=in_df.loc[:,['x','y','gene']]
    in_df=in_df.dropna()
    
    if filterGenes:
        for i in filter_genes:
            in_df=in_df[in_df['gene'].str.contains(i)==False]
            
            
    return in_df

def read_in_Vizgen(path_genes,xcol='global_x',ycol='global_y',genecol='gene',skiprows=None,offset=None, bbox=None,pixelSize=None,filterGenes=None):
    """This function read in Vizgen input data to a dask datafrmae with predefined column names. """
    
    in_df=dd.read_csv(path_genes,skiprows=skiprows)
    in_df=in_df.loc[:,[xcol,ycol,genecol]]

    in_df=in_df.rename(columns={xcol:'x',ycol:'y',genecol:'gene'})

    if bbox:
        in_df['x']-=bbox[0]
        in_df['y']-=bbox[1]
    if pixelSize:
        in_df['x'] /= pixelSize
        in_df['y'] /= pixelSize
    if offset:
            in_df['x'] -= offset[0]   
            in_df['y'] -= offset[1]
    
    in_df=in_df.dropna()
    
    if filterGenes:
        for i in filterGenes:
            in_df=in_df[in_df['gene'].str.contains(i)==False]
    return in_df


    
