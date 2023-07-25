from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def tiling_correction(
    img: np.ndarray, flatfield: np.ndarray, img_orig: np.ndarray, output: Optional[str] = None
) -> None:
    """Creates the plots based on the correction overlay and the original and corrected images."""

    # disable interactive mode
    # if output:
    #    plt.ioff()

    # Tile correction overlay
    if flatfield is not None:
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
        ax1.imshow(flatfield, cmap="gray")
        ax1.set_title("Correction performed per tile")

        # Save the plot to ouput
        if output:
            plt.close(fig1)
            fig1.savefig(output + "0.png")

    # Original and corrected image
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
    ax2[0].imshow(img, cmap="gray")
    ax2[0].set_title("Corrected image")
    ax2[1].imshow(img_orig, cmap="gray")
    ax2[1].set_title("Original image")
    ax2[0].spines["top"].set_visible(False)
    ax2[0].spines["right"].set_visible(False)
    ax2[0].spines["bottom"].set_visible(False)
    ax2[0].spines["left"].set_visible(False)
    ax2[1].spines["top"].set_visible(False)
    ax2[1].spines["right"].set_visible(False)
    ax2[1].spines["bottom"].set_visible(False)
    ax2[1].spines["left"].set_visible(False)

    # Save the plot to ouput
    if output:
        plt.close(fig2)
        fig2.savefig(output + "1.png")
