"""Define package version"""

__version__ = "0.0.1"

import os

# see geopandas https://geopandas.org/en/stable/ and https://github.com/geopandas/geopandas/releases/tag/v1.0.0
# removing this could mean only supporting gepandas >=1.0.0 and shapely 2
os.environ["USE_PYGEOS"] = "0"
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = (
    "False"  # avoid newer dataframe backends, see  https://github.com/dask/dask/issues/11146
)

loglevel = os.environ.get("LOGLEVEL")
if loglevel is None or loglevel.upper() != "DEBUG":
    # silence developer warnings if not in debug mode
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# import submodules in specific order to avoid circular imports
# use aliases from more convenient names
# isort: off
from sparrow import utils  # noqa: E402
from sparrow import io  # noqa: E402
from sparrow import datasets  # noqa: E402
from sparrow import image as im  # noqa: E402
from sparrow import plot as pl  # noqa: E402
from sparrow import points as pt  # noqa: E402
from sparrow import shape as sh  # noqa: E402
from sparrow import table as tb  # noqa: E402
# isort: on

__all__ = [
    "utils",
    "io",
    "datasets",
    "im",
    "pl",
    "pt",
    "sh",
    "tb",
    "__version__",
]
