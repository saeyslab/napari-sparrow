"""Define package version"""
__version__ = "0.0.1"

import os

os.environ["USE_PYGEOS"] = "0"

from . import image as im
from . import io, utils
from . import plot as pl
from . import shape as sh
from . import table as tb
