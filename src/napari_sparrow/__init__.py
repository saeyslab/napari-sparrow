"""Define package version"""
__version__ = "0.0.1"

import os
os.environ["USE_PYGEOS"] = "0"

from . import io
from . import image as im
from . import shape as sh
from . import table as tb
from . import plot as pl
from . import utils   # TODO do we want to keep utils? do we want to make utils actually _utils to indicate it is not guaranteed to be stable?
