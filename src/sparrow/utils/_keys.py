from enum import Enum

_CELL_INDEX = "cells"
_INSTANCE_KEY = "cell_ID"
_REGION_KEY = "fov_labels"
_ANNOTATION_KEY = "annotation"
_UNKNOWN_CELLTYPE_KEY = "unknown_celltype"
_CLEANLINESS_KEY = "Cleanliness"
_CELLSIZE_KEY = "shapeSize"

_RAW_COUNTS_KEY = "raw_counts"


# flowsom
class ClusteringKey(Enum):
    _METACLUSTERING_KEY = "metaclustering"
    _CLUSTERING_KEY = "clustering"
