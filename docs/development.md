# Development

## Setting up a development environment

First clone the GitHub repo and set it as the current directory:

```bash
git clone https://github.com/saeyslab/napari-spongepy.git
cd napari-spongepy
```

Then set up a conda virtual environment and install the plugin. For GPU support other than CUDA, comment out the `cudatoolkit` line in `environment.yml` and follow the [PyTorch](https://pytorch.org/get-started/locally/) instructions.

```bash
# or Conda
conda env create -f environment.yml
# or Mamba
mamba env update -f environment.yml --prune

conda activate napari-spongepy
pip install -e '.[testing]'
```

Use a local data folder at `data/` or point to a different location by overwriting `paths.data_dir` using the CLI (`... paths.data_dir=/srv/scratch/data/spatial/`) or using an untracked local config at `configs/local/default.yaml`:
```yaml
# @package _global_

paths:
  data_dir: /srv/scratch/data/spatial/
```

This development environment is supported for:
- Windows 11 with an NVIDIA GPU
- CentOS 7 with NVIDIA GPUs
- MacOS 12.3 with an M1 Pro

## Testing

Run the tests in the root of the project:
```
pytest
```

Do a type test:
```
mypy --ignore-missing-imports src/
```

## Automated checks

Install a pre-commit hook to run all configured checks in `.pre-commit-config.yaml`:
```
pre-commit install
pre-commit run -a
```


## Debugging

Debug in VS Code using a [Remote Attach](https://code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection) `launch.json` and debugpy:
```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ],
            "justMyCode": true
        }
    ]
}
```

```
LOGLEVEL=DEBUG python -m debugpy --listen 5678 src/pipeline.py
```