from .dnnlib import EasyDict

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def show_image_grid(imgs, ratio=0.05, plot=True, **make_grid_kwargs):
    """
    Plot grid of images
    """
    grid_img = make_grid(imgs, **make_grid_kwargs)
    fig_dims = (int(ratio * grid_img.shape[-1]), int(ratio * grid_img.shape[-2]))
    grid_img = to_pil_image(grid_img.detach())
    fig, ax = plt.subplots(figsize=fig_dims)
    ax.imshow(grid_img)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if plot:
        plt.show()