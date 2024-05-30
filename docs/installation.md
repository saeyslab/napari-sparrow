# Installation

We recommend using Anaconda to install SPArrOW, and we provide an [`environment.yml`](../environment.yml).

## 1. Create the conda environment:

```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate napari-sparrow
```

Depending on your hardware, you may need to adapt the [`environment.yml`](../environment.yml) file as follows:

- On Windows comment out the lines `basicpy==...`, `jax==...` and `jaxlib==...` in the `environment.yml`. We will install `basicpy` and `jax` manually as follows after the environment is build:

```bash
pip install "jax[cpu]===0.4.10" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy==1.0.0
```

- On Mac comment out the line `mkl=2024.0.0`.

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

## NVIDIA GPU support

We provide [environment_vib_compute.yml](../environment_vib_compute.yml) that will install `torch` with NVIDIA GPU support on Linux (tested on CentOS). After creation of the environment via `conda env create -f environment_vib_compute.yml`, activate the environment, and install `SPaRROW` via `pip install git+https://github.com/saeyslab/napari-sparrow.git`.
