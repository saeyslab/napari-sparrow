FROM condaforge/mambaforge:24.9.2-0

COPY . .

RUN mamba env create
RUN mamba run -n napari-sparrow --no-capture-output pip install .[docs]