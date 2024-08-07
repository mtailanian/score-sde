import sys
from typing import Union, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import matplotlib.patches as patches
import cv2
from scipy.optimize import curve_fit
from kornia.morphology import dilation, erosion, opening, closing
from kornia.filters import GaussianBlur2d, median, median_blur
from natsort import natsorted
import torchvision
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics.recall import Recall
from metrics.aupro import AUPRO
# from metrics.iou import IoU
from torchmetrics.classification import BinaryJaccardIndex as IoU
from metrics.mIoU import mIoU
import yaml
from skimage.restoration import denoise_nl_means
import scipy.stats as st
from PIL import Image
from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu

import datasets
# from anomaly_detector import AnomalyDetector
from path_utils import get_path
from datasets_anomaly import ALL_CATEGORIES
from utils import load_image_and_label, generate_labels_image, add_blank_row, imshow_tensor, l2
from meta_lpips import MetaLpips
from nfa_tree import compute_nfa_anomaly_score_tree

SDE = "ve"
CKPT = 10

if SDE == 've':
    from configs.ve import anomaly_256_ncsnpp_continuous as configs
elif SDE == 'subvp':
    from configs.subvp import anomaly_256_ncsnpp_continuous as configs
else:
    raise ValueError(f"Unknown SDE: {SDE}")
config = configs.get_config()
nlm_params = {
    'h': 0.8 * config.model.sigma_max,
    'sigma': config.model.sigma_max,
    'fast_mode': True,
    'preserve_range': True,
    'patch_size': 5,
    'patch_distance': 16,
    'channel_axis': 0
}


def compute_nlm(z, **kwargs):

    b, c, h, w = z.shape
    assert b == 1, "Batch size must be 1"

    z = z.to("cuda")

    patch_distance = h // 4  # Search for neighbors outside this distance

    win = 5
    hw_v = hw_h = (win - 1) // 2

    z_tiled = (
        F.pad(z, tuple([hw_h, hw_h, hw_v, hw_v]), 'reflect').
        unfold(-2, 2 * hw_v + 1, 1).
        unfold(-2, 2 * hw_h + 1, 1).
        permute(0, 2, 3, 1, 4, 5)
    )

    z_vec = z_tiled.reshape(1, h * w, c * win * win)

    # Shape [h, w, h * w]
    dists = torch.cdist(z_vec, z_vec, p=2).reshape(h, w, h * w)
    # Only allow to pick neighbors that are far away: more than patch_distance
    for i in range(h):
        for j in range(w):
            mask = torch.zeros(h, w, device=dists.device)
            mask[max(0, i - patch_distance): min(h, i + patch_distance), max(0, j - patch_distance): min(w, j + patch_distance)] = float('inf')
            mask = mask.reshape(h * w)
            dists[i, j] = dists[i, j] + mask

    n_neighbors = 10
    topk_distances, indexes = torch.topk(dists, n_neighbors, largest=False, dim=-1)
    indexes = unravel_index(indexes, (h, w))

    # Reconstruction: average of pixels
    # z_rec = torch.zeros_like(z)
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(n_neighbors):
    #             i_n, j_n = indexes[i, j, k]
    #             z_rec[:, :, i, j] = z_rec[:, :, i, j] + z[:, :, i_n, j_n]
    #             # z_rec[:, :, i, j] = z_rec[:, :, i, j] + z_tiled[:, i_n, j_n].mean(dim=(-1, -2))
    # z_rec = z_rec / n_neighbors

    z_pad = F.pad(z, tuple([hw_h, hw_h, hw_v, hw_v]), 'reflect')
    z_rec = torch.zeros_like(z_pad)
    for i in range(hw_h, hw_h+h):
        for j in range(hw_v, hw_v+w):
            total_weight = 0
            for k in range(n_neighbors):
                i_n, j_n = indexes[i - hw_v, j - hw_h, k]
                i_n, j_n = i_n + hw_v, j_n + hw_h

                h = 0.8 * config.model.sigma_max
                weight = torch.exp(-topk_distances[i - hw_v, j - hw_h, k] / h)
                total_weight += weight

                z_rec[:, :, i-hw_v:i+hw_v+1, j-hw_h:j+hw_h+1] += (weight * z_pad[:, :, i_n-hw_v:i_n+hw_v+1, j_n-hw_h:j_n+hw_h+1])
            # z_rec[:, :, i - hw_v:i + hw_v + 1, j - hw_h:j + hw_h + 1] /= total_weight
    z_rec = z_rec[:, :, hw_v:-hw_v, hw_h:-hw_h]

    imshow_tensor(l2(z), "z")
    imshow_tensor(l2(z_rec), "z_rec")
    imshow_tensor(l2(z - z_rec), "z - z_rec")

    # DEBUG scikit nlm
    z_nlm = torch.from_numpy(denoise_nl_means(z[0].cpu().numpy(), **nlm_params)).unsqueeze(0)
    imshow_tensor(l2(z_nlm), "scikit")

    # DEBUG most similar patches
    aux = l2(z)[0, 0].cpu().numpy()
    # Grid
    # pix = np.array([28, 19])
    # pix = np.array([19, 15])
    # Carpet
    # pix = np.array([40, 20])
    pix = 2 * np.array([40, 12])
    # pix = np.array([10, 10])

    fig, ax = plt.subplots()
    ax.imshow(aux)
    rect = patches.Rectangle(pix[::-1] - hw_v - 0.5, win, win, linewidth=1, edgecolor='y', facecolor='none')
    ax.add_patch(rect)
    for i, j in indexes[pix[0], pix[1]]:  # [1:]:  # [1:]:  # Exclude the pixel itself
        rect = patches.Rectangle((j - hw_v - 0.5, i - hw_v - 0.5), win, win, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

    print


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def main(dataset, category):
    # z = torch.from_numpy(np.array(Image.open(str(get_path("data") / "mvtec/zipper/test/broken_teeth/006.png")).convert("RGB").resize((256, 256)))).unsqueeze(0).permute(0, 3, 1, 2).float() / 255
    # imshow_tensor(z)
    # z_nlm = compute_nlm(z, **nlm_params)

    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"

    z_paths = natsorted(list(z_dir.glob("*.npy")))  # [::-1][:10]
    for z_path in tqdm(z_paths, desc=category):

        z = torch.from_numpy(np.load(z_path))
        # if SDE == 've':
        #     z = z / 256  # HACK!!


        # TODO: REMOVE
        z = F.interpolate(z, size=(128, 128), mode='bilinear', align_corners=False)


        z_nlm = compute_nlm(z, **nlm_params)


if __name__ == '__main__':
    dataset = "mvtec"
    category = "carpet"
    # category = "grid"
    # category = "hazelnut"
    main(dataset, category)
