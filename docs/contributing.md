# Development

## Setting up a development environment

First clone the GitHub repo and set it as the current directory:

```bash
git clone https://github.com/saeyslab/napari-sparrow.git
cd napari-sparrow
```

Then set up a conda virtual environment and install the `napari-sparrow`.

```bash
# or Conda
conda env create -f environment.yml
# or Mamba
mamba env update -f environment.yml --prune

conda activate napari-sparrow
pip install -e '.[testing,docs]'
```

This development environment is supported for:

- Windows 11 with an NVIDIA GPU
- CentOS 7 with NVIDIA GPUs
- MacOS 12.3 with an M1 Pro

## Testing

To run unit tests for the pipeline, run the following from the root of the project:

```bash
pytest src/sparrow/_tests/test_pipeline.py
```

And to run unit tests for the plugin:

```bash
pytest src/sparrow/_tests/test_widget.py
```

Continuous integration will automatically run the tests on all pull requests.

### Type testing

Do a type test:

```
mypy --ignore-missing-imports src/
```

## Automated commit checks

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
