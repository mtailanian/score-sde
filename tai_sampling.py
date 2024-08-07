import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
from scipy.optimize import curve_fit
from kornia.morphology import dilation, erosion, opening, closing
from kornia.filters import GaussianBlur2d, median, median_blur
from natsort import natsorted
import torchvision
from metrics.aupro import AUPRO
from torchmetrics.classification import BinaryJaccardIndex as IoU
from metrics.mIoU import mIoU
import yaml
from skimage.restoration import denoise_nl_means
import scipy.stats as st
from PIL import Image
from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu
from scipy import integrate
import datasets
from anomaly_detector import AnomalyDetector
from path_utils import get_path
from datasets_anomaly import ALL_CATEGORIES
from utils import load_image_and_label, generate_labels_image, add_blank_row, imshow_tensor, l2
from meta_lpips import MetaLpips
from nfa_tree import compute_nfa_anomaly_score_tree
from dad import DAD
from nfa import nfa_detection_normal, nfa_detection_binomials
from models import utils as mutils
from torchvision.utils import make_grid


SDE = 've'
CKPT = 10

if SDE == 've':
    from configs.ve import anomaly_256_ncsnpp_continuous as configs
elif SDE == 'subvp':
    from configs.subvp import anomaly_256_ncsnpp_continuous as configs


def sample(dataset, category, num_samples=8):

    config = configs.get_config()
    config.data.dataset = dataset
    config.data.category = category
    config.training.batch_size = 1

    out_path = get_path('results') / 'dad' / SDE / "samples"
    out_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(get_path('training') / 'dad' / SDE / dataset / category / 'checkpoints' / f'checkpoint_{CKPT}.pth')
    ad = AnomalyDetector(ckpt_path, config)

    samples = []
    for _ in range(num_samples):
        sample, n = ad.sampling_fn(ad.score_model)
        samples.append(sample)
    samples = torch.cat(samples, 0)
    nrow = int(np.sqrt(samples.shape[0]))
    image_grid = make_grid(samples, nrow, padding=2)
    torchvision.utils.save_image(image_grid, out_path / f"{category}.pdf")


def main():

    dataset = "mvtec"
    for category in ALL_CATEGORIES[dataset]:  # ["screw"]:  # ["carpet", "grid", "bottle"]:  # ALL_CATEGORIES[dataset]:
        sample(dataset, category, num_samples=8)


if __name__ == '__main__':
    main()
