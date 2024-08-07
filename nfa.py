import sys
from typing import Union, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
from kornia.morphology import dilation, closing

sys.setrecursionlimit(100000)


def nfa_detection_normal(z, log_nfa_thr, half_win: Union[Tuple, List, int], sigma=1., show=2, reduce='median'):
    b, c, h, w = z.shape
    if not isinstance(half_win, (tuple, list)):
        half_win = (half_win, half_win)
    wh, ww = half_win[0], half_win[1]

    z_tiled = (
        F.pad(z, tuple([ww, ww, wh, wh]), 'reflect').
        unfold(-2, 2 * wh + 1, 1).
        unfold(-2, 2 * ww + 1, 1)
    )
    if reduce == 'median':
        z_obs = z_tiled.flatten(-2, -1).abs().median(dim=-1)[0]
    elif reduce == 'max':
        z_obs = z_tiled.abs().amax(dim=(-1, -2))
    elif reduce == 'mean':
        z_obs = z_tiled.abs().mean(dim=(-1, -2))
    elif reduce == 'min':
        z_obs = z_tiled.abs().amin(dim=(-1, -2))
    else:
        print(f"Reduction {reduce} not supported. Use 'median', 'max', 'mean' or 'min'")
        raise ValueError

    # log_prob = 0.5 * ((z_max / sigma) ** 2)
    log_prob = np.log10(2) + st.norm.logsf(z_obs.abs().cpu().numpy(), scale=sigma) / np.log(10)
    log_prob = torch.from_numpy(log_prob)

    log_prob = log_prob.amin(dim=1, keepdim=True)

    log_n_tests = np.log10(h * w)
    log_nfa = log_n_tests + log_prob

    detection = log_nfa < log_nfa_thr

    # Plots
    if show >= 2:
        plt.plot(np.sort(np.unique(log_nfa.numpy())), '.')
        plt.title('logNFAs of all pixels')
        plt.grid()
        plt.show()

        plt.plot(10 ** np.sort(np.unique(log_nfa.numpy())), '.')
        plt.title('NFAs of all pixels')
        plt.grid()
        plt.show()

        plt.imshow(log_nfa[0, 0].cpu().numpy())
        plt.title('log NFA')
        plt.show()

    if show >= 1:
        plt.imshow(detection[0, 0].cpu().numpy())
        plt.title(f'Detection: log NFA < {log_nfa_thr}')
        plt.show()

    return {'log_nfa': log_nfa, 'detection': detection}


def nfa_detection_chi2(z, log_nfa_thr, half_win: Union[Tuple, List, int], show=2):
    b, c, h, w = z.shape
    if not isinstance(half_win, (tuple, list)):
        half_win = (half_win, half_win)
    wh, ww = half_win[0], half_win[1]

    log_n_tests = np.log10(h * w)

    # Log-Prob
    z2 = (z ** 2).sum(dim=1, keepdim=True)

    z2_tiled = (
        F.pad(z2, tuple([ww, ww, wh, wh]), 'reflect').
        unfold(-2, 2 * wh + 1, 1).
        unfold(-2, 2 * ww + 1, 1)
    )
    z2_max = z2_tiled.amax(dim=(-1, -2))
    z2_mean = z2_tiled.mean(dim=(-1, -2))
    z2_min = z2_tiled.amin(dim=(-1, -2))
    z2 = z2_max

    log_prob = -(c / 2) * (z2 / c - 1 - torch.log(z2 / c)) / np.log(10)
    log_prob = log_prob.amin(dim=1, keepdim=True)

    log_nfa = log_n_tests + log_prob
    detection = log_nfa < log_nfa_thr

    # Plots
    if show >= 2:
        plt.plot(10 ** np.sort(np.unique(log_nfa.numpy())), '.')
        plt.title('NFAs of all pixels')
        plt.grid()
        plt.show()

        plt.imshow(log_nfa[0, 0].cpu().numpy())
        plt.title('log NFA')
        plt.show()

    if show >= 1:
        plt.imshow(detection[0, 0].cpu().numpy())
        plt.title(f'Detection: log NFA < {log_nfa_thr}')
        plt.show()

    return detection


def chernoff_log_prob_normal(x):
    return -x ** 2 / 2 / torch.log(torch.tensor(10))


def chernoff_log_prob_chi2(x, degrees_freedom):
    return -(degrees_freedom / 2) * (x / degrees_freedom - 1 - np.log(x / degrees_freedom)) / np.log(10)


def nfa_detection_binomials(z, log_nfa_thr, n_sigmas_list=None, show=1):
    if n_sigmas_list is None:
        n_sigmas_list = [4, 3.5, 3, 2.5, 2, 1.5, 1.0]

    detection = []
    for win in [3, 5, 7, 11, 21]:
        detection.append(nfa_detection_binomial(z, win=win, log_nfa_thr=log_nfa_thr, n_sigmas_list=n_sigmas_list, show=0))
    detection = 1 * (torch.stack(detection).sum(dim=0) > 0)

    kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32).to(detection.device)
    detection_post = dilation(detection, kernel=kernel)
    detection_post = closing(detection_post, kernel=kernel)

    if show >= 1:
        plt.imshow(detection[0, 0].cpu().numpy()), plt.title(f"NFA detection with different window sizes"), plt.show()
        plt.imshow(detection_post[0, 0].cpu().numpy()), plt.title(f"NFA detection post-processed"), plt.show()

    return {'detection': detection, 'detection_post': detection_post}


def nfa_detection_binomial(z, win=21, log_nfa_thr=0, n_sigmas_list=None, show=1):
    if n_sigmas_list is None:
        n_sigmas_list = [4, 3.5, 3, 2.5, 2, 1.5, 1.0]

    half_win = np.max([int(win // 2), 1])
    n = int((2 * half_win + 1) ** 2)
    log_n_tests = np.log10(z.shape[-1] * z.shape[-2])

    log_nfa = []
    for n_sigmas in n_sigmas_list:
        p = st.norm.cdf(n_sigmas) - st.norm.cdf(-n_sigmas)
        candidates = get_gaussian_fit_candidates(
            z, show=False, n_sigmas=n_sigmas, theoretical_thr=True
        )['detection'].sum(dim=1, keepdims=True)

        if show >= 2:
            plt.imshow(candidates[0, 0].cpu().numpy()), plt.title(f"Candidates with sigma {n_sigmas}"), plt.show()

        candidates = F.pad(candidates, tuple(4 * [half_win]), 'reflect').detach().cpu()
        candidates_unfold_h = candidates.unfold(-2, 2 * half_win + 1, 1)
        candidates_unfold_hw = candidates_unfold_h.unfold(-2, 2 * half_win + 1, 1).numpy()
        observed_candidates = np.sum(candidates_unfold_hw, axis=(-2, -1))

        if show >= 2:
            plt.imshow(observed_candidates[0, 0]), plt.title(f"Observed candidates with sigma {n_sigmas}"), plt.show()

        log_nfa.append(log_n_tests + torch.tensor(st.binom.logsf(observed_candidates, n, p) / np.log(10)))

        if show >= 2:
            plt.imshow(log_nfa[-1][0, 0].cpu().numpy()), plt.title(f"NFA with sigma {n_sigmas}"), plt.show()
            plt.imshow((log_nfa[-1][0, 0] < 0).cpu().numpy()), plt.title(f"NFA detection with sigma {n_sigmas}"), plt.show()

    log_nfa = torch.stack(log_nfa).amin(dim=0)
    detection = 1 * (log_nfa < log_nfa_thr)
    if show >= 1:
        plt.imshow(log_nfa[0, 0].cpu().numpy()), plt.title(f"NFA"), plt.show()
        plt.imshow(detection[0, 0].cpu().numpy()), plt.title(f"NFA detection"), plt.show()

    return detection
