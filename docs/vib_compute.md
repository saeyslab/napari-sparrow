# High Performance Computing on VIB compute cluster.

## Set up environment

Log in to the compute cluster:

```bash
ssh -p 2022 user.name@compute.vib.be
```

Ask for an interactive session:

```bash
salloc
```

In the default environment, install support for loading environment kernels (no modules should be loaded at this time, nor conda env):

```bash
ml purge
while [ ! -z $CONDA_PREFIX ]; do conda deactivate; done

pip install environment_kernels
```

Load the Mamba module to create a conda env.

```bash
ml load Mamba
```

Next use [environment_vib_compute.yml](../environment_vib_compute.yml) to build a conda environment:

```bash
mamba env create -f environment_vib_compute.yml
```

Activate the environment:

```bash
conda activate napari-sparrow
```

Install sparrow:

```bash
pip install git+ssh://git@github.com/saeyslab/napari-sparrow.git
```

## Run spatial transcriptomics pipeline as interactive session.

Make an ipython kernel to use in a JupyterLab notebook. The displayname is what you will select in JupyterLab.

```bash
ipython kernel install --user --name napari-sparrow --display-name "napari-sparrow"
```

Now on [https://compute.vib.be](https://compute.vib.be), start a JupyterLab (check the conda environment box), and select Mamba as the system wide Conda Module, and fill in `napari-sparrow` as the name of the Custom Conda Environment.

We should now be able to run the notebook [example_spatial_vib_compute.ipynb](../experiments/example_spatial_vib_compute.ipynb) in an interactive session on the VIB compute cluster.

Input data is provided from a [RESOLVE experiment on mouse liver](https://cloud.irc.ugent.be/public/index.php/s/HrXG9WKqjqHBEzS). The dataset used in the examples is mouse liver A1-1. Please download the DAPI-stained image and the .txt file, and adopt the paths accordingly in the notebook. The marker gene list can be found [here](../experiments/markerGeneListMartinNoLow.csv).

## Usage of private S3 buckets.

Loading SpatialData objects from private S3 buckets is supported via this fork of sparrow [https://github.com/saeyslab/harpy](https://github.com/saeyslab/harpy).

Follow the instructions for setting up a conda environment as above, but rename the environment to `harpy` in [environment_vib_compute.yml](../environment_vib_compute.yml), i.e.

```
name: harpy
channels:
  - pytorch
  - conda-forge
  - conda
dependencies:
  - geopandas=0.12.2
  - leidenalg=0.9.1
  - pyqt
  ...
```

Next build the conda environment:

```bash
mamba env create -f environment_vib_compute.yml
```

Activate the environment:

```bash
conda activate harpy
```

Install sparrow from the `harpy` fork:

```bash
pip install git+ssh://git@github.com/saeyslab/harpy.git
```

Make an ipython kernel to use in a JupyterLab notebook. The displayname is what you will select in JupyterLab.

```bash
ipython kernel install --user --name harpy --display-name "harpy"
```

We should now be able to run the notebook [load_spatialdata.ipynb](../experiments/load_spatialdata.ipynb) in an interactive session on the VIB compute cluster.
