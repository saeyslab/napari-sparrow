# Installation

We recommend using Anaconda to install `SPArrOW`, and we provide an [`environment.yml`](../environment.yml).

## 1. Create the conda environment:

```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate napari-sparrow
```

If you plan to use the `SPArrOW` function `sp.im.tiling_correction`, please install `jax` and `basicpy`. On Mac and Linux, this can be done via `pip install ...`, on Windows you will have to run the following commands:

```bash
pip install "jax[cpu]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy
```

On Mac, please comment out the line `mkl=2024.0.0` in `environment.yml`.

For a mimimal list of requirements for `SPArrOW`, we refer to the [setup.cfg](../setup.cfg).

## 2. Install `SPArrOW`:

```
pip install git+https://github.com/saeyslab/napari-sparrow.git
```

## Additional dependencies

To use the plugin, run

```bash
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[plugin]"
```

To run `SPArrOW` from the `cli`:

```bash
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[cli]"
```

To be able to run the unit tests:

```bash
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[testing]"
```

## NVIDIA GPU support

We provide [environment_vib_compute.yml](../environment_vib_compute.yml) that will install `torch` with NVIDIA GPU support on Linux (tested on CentOS). After creation of the environment via `conda env create -f environment_vib_compute.yml`, activate the environment, and install `SPArrOW` via `pip install git+https://github.com/saeyslab/napari-sparrow.git`.

For VIB members we also refer to [this document](./tutorials/hpc/vib_compute.md), for an example on how to use the VIB compute cluster with GPU support.
