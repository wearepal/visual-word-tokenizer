# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    img: Image, source: torch.Tensor, vocab_size: int, patch_size: int = 16
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    cmap = generate_colormap(vocab_size)
    vis_img = np.zeros((h, w, 3))

    for i in range(vocab_size):

        mask = (source == i)
        if mask.sum() < 2:
            continue

        mask = mask.float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = np.array(cmap[i])

        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * color.reshape(1, 1, 3)

    return np.uint8(vis_img * 255)
