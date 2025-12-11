from ._allocation import allocate, bin_counts
from ._allocation_intensity import allocate_intensity
from ._annotation import cluster_cleanliness, score_genes, score_genes_iter
from ._clustering import kmeans, leiden
from ._enrichment import nhood_enrichment
from ._preprocess import preprocess_proteomics, preprocess_transcriptomics
from ._regionprops import add_regionprop_features
from ._table import add_table_layer, correct_marker_genes, filter_on_size
