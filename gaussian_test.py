from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import scipy.stats as st


def main():
    fa = 0
    N = 1000
    sigma = 1 / 5
    for _ in tqdm(list(range(N))):
        z = torch.randn((1, 3, 256, 256)) * sigma
        # log_nfa = compute_nfa_chi2(z, 1, show=False)
        log_nfa = compute_nfa_normal(z, 0, sigma=sigma, show=False)
        fa += float((log_nfa < 0).any())

    print(fa / N)
    print(N / fa)


def compute_nfa_chi2(z, half_win: Union[Tuple, List], show=True):
    b, c, h, w = z.shape
    if not isinstance(half_win, (tuple, list)):
        half_win = (half_win, half_win)
    wh, ww = half_win[0], half_win[1]

    log_n_tests = np.log10(h * w)

    # %% Log-Prob
    z2 = (z ** 2).sum(dim=1, keepdim=True)
    z2_tiled = (
        F.pad(z2, tuple([ww, ww, wh, wh]), 'reflect').
        unfold(-2, 2 * wh + 1, 1).
        unfold(-2, 2 * ww + 1, 1)
    )
    z2_min = z2_tiled.amin(dim=(-1, -2))
    log_prob = -(c / 2) * (z2_min / c - 1 - torch.log(z2_min / c)) / np.log(10)

    log_nfa = log_n_tests + log_prob

    # %% Plots
    if show:
        plt.plot(10 ** np.sort(np.unique(log_nfa.numpy())), '.')
        plt.title('NFAs of all pixels')
        plt.grid()
        plt.show()

        plt.imshow(z2[0, 0].cpu().numpy())
        plt.title('\sum_c z^2')
        plt.show()

        plt.imshow(log_nfa[0, 0].cpu().numpy())
        plt.title('log NFA')
        plt.show()

        plt.imshow((log_nfa[0, 0] < 0).cpu().numpy())
        plt.title('Detection: log NFA < 0')
        plt.show()

    return log_nfa


def compute_nfa_normal(z, half_win: Union[Tuple, List, int], sigma: [int, float] = 1, show=True):
    b, c, h, w = z.shape
    if not isinstance(half_win, (tuple, list)):
        half_win = (half_win, half_win)
    wh, ww = half_win[0], half_win[1]

    # Log-Prob
    z_tiled = (
        F.pad(z, tuple([ww, ww, wh, wh]), 'reflect').
        unfold(-2, 2 * wh + 1, 1).
        unfold(-2, 2 * ww + 1, 1)
    )
    z_min = z_tiled.amin(dim=(-1, -2))

    # log_prob = 0.5 * ((z_min / sigma) ** 2) / np.log(10)
    log_prob = np.log10(2) + st.norm.logsf(z_min.abs().cpu().numpy(), scale=sigma) / np.log(10)
    log_prob = torch.from_numpy(log_prob)

    log_prob = log_prob.amin(dim=1, keepdim=True)
    # log_prob = log_prob[:, :1]

    log_n_tests = np.log10(h * w)
    log_nfa = log_n_tests + log_prob

    # Plots
    if show:
        plt.plot(10 ** np.sort(np.unique(log_nfa.numpy())), '.')
        plt.title('NFAs of all pixels')
        plt.grid()
        plt.show()

        plt.imshow(z.abs().sum(dim=1)[0].cpu().numpy())
        plt.title('\sum_c |z|')
        plt.show()

        plt.imshow(log_nfa[0, 0].cpu().numpy())
        plt.title('log NFA')
        plt.show()

        plt.imshow((log_nfa[0, 0] < 0).cpu().numpy())
        plt.title('Detection: log NFA < 0')
        plt.show()

    return log_nfa


if __name__ == '__main__':
    main()
