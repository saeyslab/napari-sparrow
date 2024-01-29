"""Define package version"""
__version__ = "0.0.1"

import os

os.environ["USE_PYGEOS"] = "0"

subpackages = [
    "image",
    "io",
    "plot",
    "shape",
    "table",
    "utils",
]

import lazy_loader as lazy

__getattr__, __dir__, _ = lazy.attach(__name__, subpackages)
