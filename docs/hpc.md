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
hpc$ conda activate napari-sparrow
hpc$ pip install -e .
hpc$ pip install hydra-submitit-launcher
```

If you do not have your data yet on the HPC, Rsync your local data folder to the hpc and possibly symlink it to your project folder in VS Code.
```bash
local$ rsync -vzcrSLhp data/* hpc:/path/to/data/folder
hpc$ ln -s /path/to/data/folder ./
```

## HPC config

Follow the instructions as given in the [README.md](../README.md), in the section *(Hydra) CLI*. Please update the `configs/default.yaml` file in the configs folder downloaded locally (we assume this path is `/Path/to/local/configs` in the remainder of this document):


```yaml
# @package _global_

defaults:
  - override /hydra/launcher: submitit_slurm

device: cpu

paths:
  log_dir: ${oc.env:HOME}/VIB/DATA/logs/
  data_dir: ${oc.env:HOME}/VIB/DATA/

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
```bash
HYDRA_FULL_ERROR=1 sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow --cfg job
```

Run on a dataset on the login node (only for testing, no big jobs!).
If it fails on `pthread_create()` due to the CPU thread limit on login nodes, you should switch to an interactive cluster (e.g. slaking).

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow  --multirun
```

## Submit batch jobs

Tune the parameters in the local config for the right amount of computational resources.
You have to pass the `--multirun` flag when submitting a job. For more info, see the documentation of the [Hydra Submitit Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/).


## Organize all experiment configs

You can organize different experiments using the [Experiment pattern](https://hydra.cc/docs/patterns/configuring_experiments/).

See `/Path/to/local/configs/experiment/resolve_liver.yaml` for an example:

```yaml
# @package _global_

defaults:
  - override /dataset: resolve_liver
  - override /segmentation: cellpose_resolve_liver
  - override /paths: output
  - override /clean: resolve_liver
  - override /allocate: resolve_liver
  - override /annotate: resolve_liver
  - override /visualize: resolve_liver
```

You can now run this experiment from the command line:

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow  --multirun
```


You can create new datasets by adding a config *.yaml* to your local `/Path/to/local/configs/dataset/` folder. Extend existing configs using the defaults keyword. Make sure that the name of the new dataset is unique:

```yaml
# e.g. configs/dataset/resolve_liver_2.yaml

defaults:
  - dataset/resolve_liver

image: ${dataset.data_dir}/20272_slide1_A1-2_DAPI.tiff
coords: ${dataset.data_dir}/20272_slide1_A1-2_results.txt
```

Use this newly defined dataset in a new experiment configuration at `/Path/to/local/configs/experiment/resolve_liver_2.yaml`.

```yaml
# @package _global

defaults:
  - override /dataset: resolve_liver_2
  - override /segmentation: cellpose_resolve_liver
  - override /paths: output
  - override /clean: resolve_liver
  - override /allocate: resolve_liver
  - override /annotate: resolve_liver
  - override /visualize: resolve_liver
```

Now running the experiment can be done using only the name of the experiment:

```bash
sparrow +experiment=resolve_liver_2 hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow  --multirun
```

By creating multiple experiment configs, experiments can be cleanly defined by composing existing configs and overwriting only a few new parameters.

## Submit multiple experiments

Multiple experiment can be run using a comma:

```bash
sparrow +experiment=resolve_liver,resolve_liver_2 hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow  --multirun
```

## Use of configuration files generated by the napari-sparrow plugin

After every step of the pipeline in the napari-sparrow plugin, a corresponding configuration *.yaml* file is saved in the `configs` folder of the chosen output directory containing the tuned parameters. These *.yaml* files can be used to rerun the same experiment via the CLI. This is useful when the parameters are tuned on a small crop of the image, and the user wants to use these parameters on the complete image in an HPC environment, or one the CLI on a more powerful workstation

To use these generated *.yaml* files, please copy them to the corresponding directory of the configs folder downloaded locally (i.e. the configs folder that is in the hydra searchpath). E.g., the generated *.yaml* `configs/clean/plugin.yaml` should be placed in the corresponding `/Path/to/local/configs/clean` folder in the hydra searchpath. 

In a similar way as above, one should then create a new experiment configuration, e.g. `/Path/to/local/configs/experiment/plugin.yaml`:

```yaml
# @package _global

defaults:
  - override /dataset: resolve_liver
  - override /segmentation: plugin
  - override /paths: output
  - override /clean: plugin
  - override /allocate: plugin
  - override /annotate: plugin
```

An experiment can then be started as follows:

```bash
sparrow +experiment=plugin hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow  --multirun
```

Note that if parameters were tuned on a crop of the image (`clean.crop_param` and `segmentation.crop_param` in the corresponding *.yaml*), you must set these parameters to `null`, if you want to run on the uncropped image.