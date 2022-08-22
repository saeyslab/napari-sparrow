#!/bin/bash

while getopts 'sph' OPTION; do
  case "$OPTION" in
    s)
      echo "run notebooks in serial"
      pytest --nbmake ../../../experiments --ignore=experiments\create_xarray_zarr.ipynb --ignore=experiments\example_hydra.ipynb --ignore=experiments\read_vizgen.ipynb --ignore=experiments\scipy-edt\scipy-distance-transform-experiment.ipynb
      ;;
    p)
      echo "run notebooks in parallel"
      pytest --nbmake -n=auto ../../../experiments --ignore=experiments\create_xarray_zarr.ipynb --ignore=experiments\example_hydra.ipynb --ignore=experiments\read_vizgen.ipynb --ignore=experiments\scipy-edt\scipy-distance-transform-experiment.ipynb
      ;;
    h)
      echo -e "Help \nTesting the notebooks with pytest, run them in serial with -s or in parallel with the -p option"
      ;;
    ?)
      echo "script usage: $(basename \$0) [-p] [-s] [-h] " >&2
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"