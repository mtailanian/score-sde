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
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics.recall import Recall
from metrics.aupro import AUPRO
from torchmetrics.classification import BinaryJaccardIndex as IoU
from metrics.mIoU import mIoU
import yaml
from skimage.restoration import denoise_nl_means
import scipy.stats as st
from PIL import Image
from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu
from torchvision.utils import save_image
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
from skimage.restoration import estimate_sigma, denoise_nl_means

sys.setrecursionlimit(100000)


CKPT = 10
SDE = 've'
# SDE = 'subvp'

if SDE == 've':
    from configs.ve import anomaly_256_ncsnpp_continuous as configs
elif SDE == 'subvp':
    from configs.subvp import anomaly_256_ncsnpp_continuous as configs


def dequantize(image):
    return (torch.rand_like(image) + image * 255.) / 256.


def get_hutchinson_epsilon(image, hutchinson_type='Rademacher'):
    if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(image)
    elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(image, low=0, high=2).float() * 2 - 1.
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    return epsilon


def to_tensor(x, shape, device):
    return torch.from_numpy(x.reshape(shape)).to(device).type(torch.float32)


def to_flat_np(x):
    return x.detach().cpu().numpy().reshape((-1,))


def z_prior_logp(z, sde):
    if SDE == 've':
        N = np.prod(z.shape[1])  # number of channels
        return (
                -N / 2. * np.log(2 * np.pi * sde.sigma_max ** 2)
                - torch.sum(z ** 2, dim=1, keepdim=True) / (2 * sde.sigma_max ** 2)
        )


def z_prior_logp_single_channel(z, sde):
    if SDE == 've':
        return -1 / 2. * np.log(2 * np.pi * sde.sigma_max ** 2) - (z ** 2) / (2 * sde.sigma_max ** 2)


def div_f_hutch(score_model, sde, x, t, epsilon):

    def drift_fn(xx, tt):
        score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        f_tilde = rsde.sde(xx, tt)[0]  # f - g^2 * score
        return f_tilde

    with torch.enable_grad():
        x.requires_grad = True
        f = drift_fn(x, t)
        f_eps = (f * epsilon).sum()
        grad_f_eps = torch.autograd.grad(f_eps, x)[0]
    x.requires_grad = False

    div_f = grad_f_eps * epsilon
    return f, div_f


def compute_likelihood(score_model, sde, image, eval_type='u'):
    shape = image.shape
    b, c, h, w = shape
    device = image.device

    epsilon = get_hutchinson_epsilon(image)

    def ode(t, x):
        _f = to_tensor(x[:-shape[0]], shape, device)  # drift
        _div_f = x[-shape[0]:]  # logp_grad

        vec_t = torch.ones(b, device=device) * t
        f, div_f = div_f_hutch(score_model, sde, _f, vec_t, epsilon)

        return np.concatenate([
            to_flat_np(f),
            to_flat_np(div_f.sum(dim=(1, 2, 3)))
        ], axis=0)

    # TODO: differences with i-DODE
    #   uses range between -1 and 1
    #   t_span goes from gamma_0 = -11.8 (remark A.2) and gamma(1)

    logps = []
    num_importance_sampling = 1
    if eval_type == 'u':  # Uniform dequantization

        for _ in tqdm(list(range(num_importance_sampling)), desc="Importance sampling"):
            image_dequantized = dequantize(image)

            init = np.concatenate([to_flat_np(image_dequantized), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode, (1e-5, sde.T), init, rtol=1e-5, atol=1e-5, method='RK45')

            num_time_steps = solution.y.shape[1]
            ode_trajectories = to_tensor(solution.y[:-shape[0], :], [*shape, num_time_steps], device="cpu")

            z = ode_trajectories[..., -1]

            # prior_logp = z_prior_logp(z, sde)
            prior_logp = z_prior_logp_single_channel(z, sde)

            # delta_logp = to_tensor(solution.y[-shape[0]:, -1], (shape[0],), "cpu")
            vec_T = torch.ones(b, device=device) * sde.T
            delta_logp = div_f_hutch(score_model, sde, z.detach().to(device), vec_T, epsilon)[1].detach().cpu().requires_grad_(False)

            #
            #
            #
            # delta_logps = []
            # for index, tt in enumerate(solution.t):
            #     vec_tt = torch.ones(b, device=device) * tt
            #     delta_logps.append(div_f_hutch(score_model, sde, ode_trajectories[..., index].detach().to(device), vec_tt, epsilon)[1].detach().cpu().requires_grad_(False))
            # delta_logp_integral = torch.trapz(torch.stack(delta_logps), torch.from_numpy(solution.t), dim=0)
            #
            #

            logp = prior_logp + delta_logp
            logps.append(logp)

            imshow_tensor(image)
            imshow_tensor(prior_logp.sum(dim=1), f"prior logp [{torch.quantile(prior_logp, .001):0.2f}, {torch.quantile(prior_logp, .999):0.2f}]")
            imshow_tensor(delta_logp.sum(dim=1), f"delta logp [{torch.quantile(delta_logp, .001):0.2f}, {torch.quantile(delta_logp, .999):0.2f}]")
            imshow_tensor(logp.sum(dim=1), f"logp [{torch.quantile(logp, .001):0.2f}, {torch.quantile(logp, .999):0.2f}]")
            print

    else:
        raise NotImplementedError(f"Evaluation type {eval_type} unknown.")
    # elif eval_type == 'v':  # Test by variational
    #     pass

    logps = torch.stack(logps, dim=0)
    log_likelihood = l2(logps.mean(dim=0))
    imshow_tensor(log_likelihood)

    # sigma_est = np.mean(estimate_sigma(logp[0].cpu().numpy(), channel_axis=0))
    sigma_est = 348
    logp_nlm = torch.from_numpy(denoise_nl_means(
        logp[0].cpu().numpy(),
        h=0.8 * sigma_est, sigma=sigma_est,
        fast_mode=False, preserve_range=True,
        patch_size=9, patch_distance=16, channel_axis=0,
    )).unsqueeze(0)
    imshow_tensor(l2(logp_nlm))

    ll = log_likelihood - log_likelihood.min()
    ll = ll / ll.max()
    save_image(ll, "results/ll.pdf")

    return log_likelihood


def compute(dataset, category):

    config = configs.get_config()
    config.data.dataset = dataset
    config.data.category = category
    config.training.batch_size = 1

    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    # ckpt_path = str(get_path('training') / 'dad' / SDE / dataset / category / 'checkpoints' / f'checkpoint_{CKPT}.pth')
    ckpt_path = str(get_path('training') / "tmp/ve/mvtec/carpet/exp_0012/ckpts/TRAIN_6122_loss-1599.85583.ckpt")
    ad = DAD(config)
    ad.load_state_dict(torch.load(ckpt_path)["state_dict"])

    # it = iter(eval_ds)
    # for _ in range(20):
    #     batch = next(it)
    # l = compute_likelihood(ad.score_model, ad.sde, batch['image'].to(config.device))

    for batch in tqdm(eval_ds, desc=category):
        l = compute_likelihood(ad.score_model, ad.sde, batch['image'].to(config.device))


def main():

    dataset = "mvtec"
    for category in ALL_CATEGORIES[dataset][0:]:  # ["screw"]:  # ["carpet", "grid", "bottle"]:  # ALL_CATEGORIES[dataset]:
        compute(dataset, category)


if __name__ == '__main__':
    main()
