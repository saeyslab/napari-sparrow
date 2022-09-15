# High Performance Computing

Running in a cluster environment is complicated.
We support the HPC environment of [VSC](https://www.ugent.be/hpc/en).
It uses [EasyBuild](https://docs.easybuild.io/en/latest/) for software installation and [SLURM](https://slurm.schedmd.com/documentation.html) for workload managment.

For basic HPC setup, follow the [manual](https://www.ugent.be/hpc/en/support/documentation.htm) of your HPC administration.

## Setup your environment on the cluster

Clone this repository. You can work remotely on the login node using the VS Code Remote plugin.
We use a combination of EasyBuild modules and [Mamba](https://github.com/conda-forge/miniforge).
It is best to start working on an interactive debugging cluster and submit large jobs on bigger non-interactive clusters.

```bash
hpc$ ml switch cluster/slaking
hpc$ ml load CUDA
hpc$ mamba env update -f environment.yml
hpc$ conda activate napari-spongepy
hpc$ pip install -e .
hpc$ pip install hydra-submitit-launcher
```

If you do not have your data yet on the HPC, Rsync your local data folder to the hpc and possibly symlink it to your project folder in VS Code.
```bash
local$ rsync -vzcrSLhp data/* hpc:/path/to/data/folder
hpc$ ln -s /path/to/data/folder ./
```

## HPC config

Add the following file at `configs/default.yaml`. Make sure to overwrite using the correct paths.
`paths.log_dir` should be a large storage location for saving all the logs and `paths.data_dir` should be your data folder.

```yaml
# @package _global_

defaults:
  - override /hydra/launcher: submitit_slurm

device: cpu

paths:
  log_dir: ${oc.env:VSC_DATA_VO_USER}/logs/
  data_dir: ${oc.env:VSC_DATA_VO}/spatial/

hydra:
  launcher:
    submitit_folder: ${hydra.sweep.dir}/.submitit/%j
    timeout_min: 60
    cpus_per_task: 2
    gpus_per_node: 0
    tasks_per_node: 1
    mem_gb: 4
    nodes: 1
    name: ${hydra.job.name}
```

## Run a test job locally

Check the configuration has no errors:
```
HYDRA_FULL_ERROR=1 spongepy --cfg job
```

Run on a dataset on the login node (only for testing, no big jobs!).
If it fails on `pthread_create()` due to the CPU thread limit on login nodes, you should switch to an interactive cluster (e.g. slaking).
```
spongepy +dataset=resolve_liver
```

## Submit batch jobs

Tune the parameters in the local config for the right amount of computational resources.
You have to pass the `--multirun` flag when submitting a job. For more info, see the documentation of the [Hydra Submitit Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/).
```
spongepy +dataset=resolve_liver subset=\'0:100,0:100\' +segmentation=watershed --multirun
```

## Organize all experiment configs

You can organize different experiments using the [Experiment pattern](https://hydra.cc/docs/patterns/configuring_experiments/).

You can create new datasets by adding a config .yaml to your local `configs/dataset/` folder. Extend existing configs using the defaults keyword. Make sure that the name of the new dataset is unique.

```yaml
# e.g. configs/dataset/resolve_liver2.yaml

defaults:
  - dataset/resolve_liver

data_dir: ${paths.data_dir}/resolve_liver2
dtype: tiff
image: ${dataset.data_dir}/33075-974_A2-1_DAPI.tiff
coords: ${dataset.data_dir}/33075-974_A2-1_results.txt
brightfield: ${dataset.data_dir}/33075-974_A2-1_brightfield.tiff
raw: ${dataset.data_dir}/33075-974_A2-1_raw.tiff
markers: null
```

Use this newly defined dataset in a new experiment configuration at `config/experiment/liver_small.yaml`.

```yaml
# @package _global

defaults:
  - override /dataset: resolve_liver2

subset: '0:2144,0:2144'
```

Now running the experiment can be done using only the name of the experiment:
```
spongepy +experiment=liver_small
```

By creating multiple experiment configs, experiments can be cleanly defined by composing existing configs and overwriting only a few new parameters.

## Submit multiple experiments

Multiple experiment can be run using a comma or the [glob syntax]().
```bash
spongepy +experiment=liver_small,brain_small -m
```

For large batches, create the an extra folder of configs programmatically using `scripts/create_dataset.py` based on the folder structure in the dataset.
```bash
python scripts/create_dataset.py -f /srv/scratch/data/spatial/resolve_melanoma -c configs/ms_melanoma/dataset
```

You can then run all are part of the datasets by pointing to this additional config folder.
```bash
spongepy -cd configs/ms_melanoma dataset='glob(*A1*)' -m
```

Alternatively, you can supply argmuments in the commandline using bash functions like find, sed and paste:
```bash
spongepy dataset=$(find src/napari_spongepy/configs/dataset/multisample_resolve_melanoma  -name '*.yaml' | sed -E 's@.*/(.*/.*)$@\1@g' | paste -sd ',' -) paths=output -m
```
