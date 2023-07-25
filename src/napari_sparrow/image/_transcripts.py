import numpy as np
from scipy.ndimage import gaussian_filter         # FIXME: which gaussian_filter?
from scipy.ndimage.filters import gaussian_filter # FIXME: which gaussian_filter?


def transcript_density(df, scaling_factor=100):  # TODO: add type hints
    """This function plots the transcript density of the tissue. You can use it to compare different regions in your tissue on transcript density."""
    Try = df.groupby(["x", "y"]).count()["gene"]  # TODO: rename Try + lower case
    image = np.array(Try.unstack(fill_value=0))
    image = image / np.max(image)
    blurred = gaussian_filter(scaling_factor * image, sigma=7)
    return blurred.T
