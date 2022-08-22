#### Testing
### Testing the pipeline
The test suites are executed by Pytest, in order to execute all the tests in the _test folder, type ```pytest``` in your cli.

### Testing the notebooks
By using the plugin nbmake for pytest, you can test all the notebooks in the experiments folder. 
Execute with: ```pytest  --nbmake experiments --ignore=experiments\create_xarray_zarr.ipynb --ignore=experiments\example_hydra.ipynb --ignore=experiments\read_vizgen.ipynb --ignore=experiments\scipy-edt\scipy-distance-transform-experiment.ipynb``` or execute in parallel with the pytest-xdist package via the command: ```pytest --nbmake -n=auto experiments --ignore=experiments\create_xarray_zarr.ipynb --ignore=experiments\example_hydra.ipynb --ignore=experiments\read_vizgen.ipynb --ignore=experiments\scipy-edt\scipy-distance-transform-experiment.ipynb```