# Installation

There are two different installation methods:

1. Using the Python package manager pip (TODO)
```
pip install napari-sparrow
```

2. Installation from source

First clone this GitHub repo and set it as the current directory:
```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

Depending on your hardware, you may need to adapt the Conda `environment.yml` file as follows:
- On Windows comment out the line `basicpy==1.0.0`. We will install `basicpy` manually, see below.

Now create the conda environment
```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate napari-sparrow
```

On Windows one must manually install `basicpy` and `jax` as follows:
```
pip install "jax[cpu]===0.4.10" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy==1.0.0
```

Finally, install `napari-sparrow`
```
pip install -e .
```

## Additional dependencies

To use the plugin, run

```bash
pip install "napari-sparrow[plugin]"
```

or when build from source:

```bash
pip install -e ".[plugin]"
```

To run `sparrow` from the `cli`:

```bash
pip install "napari-sparrow[cli]"
```

or when build from source:

```bash
pip install -e ".[cli]"
```

