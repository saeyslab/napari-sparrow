#!/usr/bin/env bash

micromamba run -n napari-sparrow python -m sphinx -T -b html -d _build/doctrees -D language=en docs docs/html
