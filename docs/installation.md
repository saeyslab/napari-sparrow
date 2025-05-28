# Installation

We recommend using a virtual environment such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [mamba](https://mamba.readthedocs.io/en/latest/index.html), [venv](https://docs.python.org/3/library/venv.html) or others. Instructions below are for conda, but are completely analogous for other solutions.


## 1. Create the environment
Create a and activate new environment with Python 3.10:

```bash
conda create -n napari-sparrow python=3.10 -c conda-forge
conda activate napari-sparrow
```

## 2. Install `SPArrOW`
Installing SPArrOW can be done by cloning the repository and installing locally:

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
pip install .
```

Alternatively, `pip` can install `SPArrOW` directly from github as follows:

```bash
pip install git+https://github.com/saeyslab/napari-sparrow.git
```

### Optional dependencies
`SPArrOW` includes a number of optional dependencies for specific use cases.
These are listed below.

To use the function `sp.im.tiling_correction`:
```bash
pip install .[tiling]
# alternatively:
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[tiling]"
```

To use the Napari plugin:
```bash
pip install .[plugin]
# alternatively:
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[plugin]"
```

To run `SPArrOW` from the `cli`:

```bash
pip install .[cli]
# alternatively:
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[cli]"
```

To be able to run the unit tests:

```bash
pip install .[testing]
# alternatively:
pip install "git+https://github.com/saeyslab/napari-sparrow.git#egg=sparrow[testing]"
```
