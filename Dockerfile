FROM sphinxdoc/sphinx

WORKDIR /napari-sparrow
COPY . /napari-sparrow

RUN pip install .[docs]

CMD ["python", "-m", "sphinx", "-T", "-b", "html", "-d", "_build/doctrees", "-D", "language=en", "docs", "_build/html"]