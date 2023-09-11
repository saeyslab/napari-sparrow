# Use Ubuntu 20.04 as base image
FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=non-interactive

RUN apt-get update && apt-get install -y wget bzip2 libarchive13 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH="/root/miniconda/bin:$PATH"

# Install mamba using conda
RUN conda install mamba -c conda-forge

# complex dependencies that needs to be solved with conda
RUN mamba install -c conda-forge gcc libgdal gxx imagecodecs -y

COPY environment.yml /environment.yml

COPY . /napari-sparrow

# Install the conda environment
RUN mamba env create -f /environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "napari-sparrow", "/bin/bash", "-c"]

WORKDIR /napari-sparrow
RUN pip install -e '.[testing,cli]'