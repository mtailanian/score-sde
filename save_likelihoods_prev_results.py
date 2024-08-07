from typing import Union, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import curve_fit
import scipy.stats as st
from kornia.morphology import opening, closing
from natsort import natsorted
import torchvision
from ignite.metrics.recall import Recall
from metrics.mIoU import mIoU
import yaml

import datasets
# from anomaly_detector import AnomalyDetector
from configs.ve import anomaly_256_ncsnpp_continuous as configs
from path_utils import get_path
from datasets_anomaly import ALL_CATEGORIES
from utils import load_image_and_label, generate_labels_image, add_blank_row

CKPT = 10


def save_z(category):
    # ckpt_path = ckpts[category]
    ckpt_path = str(get_path('training') / 'dad' / category / 'checkpoints' / f'checkpoint_{CKPT}.pth')

    out_dir = get_path('results') / "dad" / "mvtec" / category / f"ckpt_{CKPT}" / "z"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = configs.get_config()
    config.data.dataset = "mvtec"
    config.data.category = category
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    ad = AnomalyDetector(ckpt_path)
    for batch in tqdm(eval_ds, desc=category):
        bpd, z, nfe = ad.likelihood_fn(ad.score_model, batch['image'].to(config.device))

        for i, zi in enumerate(z):
            img_path = Path(batch['image_path'][i])
            defect_type = img_path.parents[0].name
            output_path = out_dir / f"{defect_type}_{img_path.stem}.npy"
            np.save(output_path, zi.unsqueeze(0).cpu().numpy())


def save_likelihood_masks_grid(category, out_size=128):

    out_path = get_path('results') / "dad" / "likelihood_masks" / f"ckpt_{CKPT}"
    out_path.mkdir(parents=True, exist_ok=True)

    resize_transform = torchvision.transforms.Resize(out_size, antialias=True)
    z_dir = get_path('results') / "dad" / "mvtec" / category / f"ckpt_{CKPT}" / "z"

    all_images = {
        'original': [],
        'label': [],
        'z': [],
        'gauss_3_sigmas': [],
        'gauss_3_sigmas_post': [],
        'gauss_3_sigmas_theo': [],
        'gauss_3_sigmas_theo_post': [],
        'gauss_2_sigmas': [],
        'gauss_2_sigmas_post': [],
        'nfa_binomial_win-3': [],
        'nfa_thr-0_half-win-0': [],
        'nfa_thr-1_half-win-0': [],
        'nfa_thr-0_half-win-1': [],
        'nfa_thr-1_half-win-1': [],
        'DBSCAN': [],
        'IsolationForest': [],
        'EllipticEnvelope': [],
    }
    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):

        data = load_image_and_label(z_path, out_size)
        all_images['original'].append(data['image_with_label'] / 255)
        all_images['label'].append(data['label'] / 255)

        # HACK: / 256
        z = torch.from_numpy(np.load(z_path)) / 256

        l2 = (z ** 2).sum(dim=1, keepdims=True).sqrt()
        l2 -= l2.min()
        l2 /= l2.quantile(.99)
        all_images['z'].append(resize_transform(l2)[0])

        # Gaussian Fit
        for n_sigmas in [3, 2]:
            gauss_fit_results = get_gaussian_fit_candidates(z, n_sigmas=n_sigmas, show=False)
            all_images[f'gauss_{n_sigmas}_sigmas'].append(resize_transform(gauss_fit_results['candidates'][0]))
            all_images[f'gauss_{n_sigmas}_sigmas_post'].append(resize_transform(gauss_fit_results['candidates_post'][0]))
            if n_sigmas == 3:
                gauss_fit_results = get_gaussian_fit_candidates(z, n_sigmas=n_sigmas, show=False, theoretical_thr=True)
                all_images[f'gauss_{n_sigmas}_sigmas_theo'].append(resize_transform(gauss_fit_results['candidates'][0]))
                all_images[f'gauss_{n_sigmas}_sigmas_theo_post'].append(resize_transform(gauss_fit_results['candidates_post'][0]))

        # NFA
        all_images[f'nfa_binomial_win-3'].append(resize_transform(nfa_detection_binomial(z, win=3, show=0)[0]))
        all_images[f'nfa_thr-{0}_half-win-{0}'].append(resize_transform(nfa_detection_chi2(z, log_nfa_thr=0, half_win=0, show=0)[0]))
        all_images[f'nfa_thr-{1}_half-win-{0}'].append(resize_transform(nfa_detection_chi2(z, log_nfa_thr=1, half_win=0, show=0)[0]))
        all_images[f'nfa_thr-{0}_half-win-{1}'].append(resize_transform(nfa_detection_chi2(z, log_nfa_thr=0, half_win=1, show=0)[0]))
        all_images[f'nfa_thr-{1}_half-win-{1}'].append(resize_transform(nfa_detection_chi2(z, log_nfa_thr=1, half_win=1, show=0)[0]))

        # Clustering
        clustering_det = clustering_detection(z, show=0)
        for k in clustering_det.keys():
            all_images[k].append(resize_transform(clustering_det[k][0]))

    grid_keys = [k for k in all_images.keys() if k != 'label']
    metric_keys = [k for k in all_images.keys() if k not in ['original', 'label', 'z']]

    # %% Extract metrics
    all_mious = {key: mIoU(thresholds=0.5) for key in metric_keys}
    all_recalls = {key: Recall() for key in metric_keys}
    for k in metric_keys:
        for i in range(len(z_paths)):
            all_mious[k].update((all_images[k][i] > 0.5).float().ravel(), (all_images['label'][i] > 0.5).float().ravel())
            all_recalls[k].update(((all_images[k][i] > 0.5).float().ravel(), (all_images['label'][i] > 0.5).float().ravel()))

    # Compute
    out_metrics = {}
    for k in metric_keys:
        recall = all_recalls[k].compute().item()
        miou = all_mious[k].compute().item()
        print(f"{k}: mIoU={miou:.3f}, Recall={recall:.3f}")
        out_metrics[k] = {'mIoU': miou, 'Recall': recall}

    # Save
    with open(str(out_path / f"metrics_{category}.yaml"), 'w') as yaml_file:
        yaml.dump(out_metrics, yaml_file, default_flow_style=False)

    # %% Generate final grid image
    for k in grid_keys:
        all_images[k] = torch.stack(all_images[k])
        if all_images[k].shape[1] == 1:
            all_images[k] = all_images[k].repeat(1, 3, 1, 1)

    # Generate final image
    out_img = torchvision.utils.make_grid(
        torch.cat([all_images[k] for k in grid_keys], dim=0),
        normalize=False, nrow=len(z_paths), padding=2, pad_value=255
    )
    # Add labels
    labels_image = generate_labels_image(list(grid_keys), out_size, padding=2, font_scale=0.8)
    out_img = torch.concat([labels_image, out_img], dim=2)

    # Add blank rows
    out_img = add_blank_row(out_img, [2, 8, 9, 13], 20, out_size, padding=2)

    # plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.show()

    torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")


def print_likelihood_masks_metrics():
    out_path = get_path('results') / "dad" / "likelihood_masks" / f"ckpt_{CKPT}"
    metrics = {}
    for category in ALL_CATEGORIES['mvtec']:
        with open(str(out_path / f"metrics_{category}.yaml"), 'r') as yaml_file:
            metrics[category] = yaml.safe_load(yaml_file)

    sep = " "
    methods = metrics[ALL_CATEGORIES['mvtec'][0]].keys()
    print(f"{'Category'}{sep}{(sep + sep).join([k for k in methods])}")
    print(f"-{sep}{sep.join([f'mIoU{sep}Recall' for k in methods])}")
    for category in ALL_CATEGORIES['mvtec']:
        print(f"{category}", end=sep)
        for method in methods:
            print(f"{100 * metrics[category][method]['mIoU']:.2f}{sep}{100 * metrics[category][method]['Recall']:.2f}", end=sep)
        print()


def save_final_likelihood_masks(category, out_size=128):
    z_dir = get_path('results') / "dad" / "mvtec" / category / f"ckpt_{CKPT}" / "z"
    mask_dir = get_path('results') / "dad" / "mvtec" / category / f"ckpt_{CKPT}" / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):

        # HACK: / 256
        z = torch.from_numpy(np.load(z_path)) / 256

        gauss_mask_1 = get_gaussian_fit_candidates(z, n_sigmas=2, show=False)['candidates_post'][0]
        gauss_mask_2 = get_gaussian_fit_candidates(z, n_sigmas=3, show=False)['candidates'][0]
        nfa_mask_1 = nfa_detection_chi2(z, log_nfa_thr=1, half_win=1, show=0)[0]
        nfa_mask_2 = nfa_detection_chi2(z, log_nfa_thr=0, half_win=1, show=0)[0]

        # Save
        masks = torch.stack([gauss_mask_1, gauss_mask_2, nfa_mask_1, nfa_mask_2])
        np.save(mask_dir / z_path.name, masks.cpu().numpy())


def test_likelihood_masks(category):
    z_dir = get_path('results') / "dad" / "mvtec" / category / f"ckpt_{CKPT}" / "z"

    for z_path in tqdm(natsorted(list(z_dir.glob("*.npy")))):
        z_orig = torch.from_numpy(np.load(z_path))

        # HACK!!
        z = z_orig / 256

        l1 = z.abs().sum(dim=1, keepdims=True)
        l22 = (z ** 2).sum(dim=1, keepdims=True)
        l2 = torch.sqrt(l22)

        # plt.imshow(l1[0, 0].cpu().numpy()), plt.show()
        plt.imshow(l2[0, 0].cpu().numpy()), plt.title('z/256: L2'), plt.show()
        # plt.imshow(l22[0, 0].cpu().numpy()), plt.show()

        #
        #
        # sigma_est = np.mean(estimate_sigma(z[0], channel_axis=0))
        # print(f'estimated noise standard deviation = {sigma_est}')
        # patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=0)  # 5x5 patches  # 13x13 search area
        # # denoise = denoise_nl_means(z[0].numpy(), h=1.15 * sigma_est, fast_mode=False, **patch_kw)
        # # denoise2 = denoise_nl_means(z[0].numpy(), h=0.8 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw)
        # denoise3 = denoise_nl_means(z[0].numpy(), h=0.8, sigma=1, fast_mode=False, preserve_range=True, **patch_kw)
        # z = torch.from_numpy(denoise3).unsqueeze(0)
        # plt.imshow(np.sqrt((denoise3 ** 2).sum(axis=0))), plt.title('z denoised: L2'), plt.show()
        #
        # _ = get_gaussian_fit_candidates(z, n_sigmas=3, show=True, theoretical_thr=True)
        # _ = get_gaussian_fit_candidates(z, n_sigmas=2, show=True, theoretical_thr=True)
        # _ = nfa_detection_chi2(z, log_nfa_thr=1, half_win=2, show=2)
        # print
        #
        #

        # lik = gauss_func(z, 1, 0, 1)
        # plt.imshow(lik[0, 0].cpu().numpy()), plt.show()
        # plt.imshow(lik[0, 1].cpu().numpy()), plt.show()

        likelihood = torch.exp(-torch.mean(z ** 2, dim=1, keepdim=True) * 0.5)
        plt.imshow(likelihood[0, 0].cpu().numpy()), plt.title('Likelihood'), plt.show()
        # likelihood = torch.exp(-torch.sum(z ** 2, dim=1, keepdim=True) * 0.5)
        # plt.imshow(likelihood[0, 0].cpu().numpy()), plt.show()

        log_nfa_thr = 0

        candidates11 = get_gaussian_fit_candidates(z, n_sigmas=3, show=True)
        candidates12 = get_gaussian_fit_candidates(z, n_sigmas=2, show=True)
        candidates2 = nfa_detection_binomial(z, win=3, show=2)
        candidates3 = nfa_detection_chi2(z, log_nfa_thr, half_win=0, show=2)
        candidates4 = clustering_detection(z, show=1)

        for thr in range(0, 4):
            _ = nfa_detection_chi2(z, thr, half_win=0, show=1)
        for thr in range(0, 4):
            _ = nfa_detection_chi2(z, thr, half_win=1, show=1)
        for thr in range(0, 4):
            _ = nfa_detection_chi2(z, thr, half_win=2, show=1)

        print


def clustering_detection(z, show=1):
    z2_sum = (z[0] ** 2).sum(dim=0)
    np_diffs = z2_sum.view(-1, 1).detach().cpu().numpy()

    candidates = {}

    from sklearn.cluster import DBSCAN
    clf = DBSCAN()
    clf.fit(np_diffs)
    candidates['DBSCAN'] = torch.tensor(clf.labels_, device=z.device).view(1, 1, 256, 256) < 0
    if show >= 1:
        plt.imshow(candidates['DBSCAN'][0, 0].cpu().numpy()), plt.title('DBSCAN'), plt.show()

    from sklearn.ensemble import IsolationForest
    clf = IsolationForest()
    clf.fit(np_diffs)
    candidates['IsolationForest'] = torch.tensor(clf.predict(np_diffs)).view(1, 1, 256, 256) < 0
    if show >= 1:
        plt.imshow(candidates['IsolationForest'][0, 0].cpu().numpy()), plt.title('IsolationForest'), plt.show()

    # from sklearn.neighbors import LocalOutlierFactor
    # clf = LocalOutlierFactor(n_neighbors=5)
    # candidates['LocalOutlierFactor'] = torch.tensor(clf.fit_predict(np_diffs)).view(1, 1, 256, 256) < 0
    # if show >= 1:
    #     plt.imshow(candidates['LocalOutlierFactor'][0, 0].cpu().numpy()), plt.title('LocalOutlierFactor'), plt.show()

    from sklearn.covariance import EllipticEnvelope
    clf = EllipticEnvelope()
    clf.fit(np_diffs)
    candidates['EllipticEnvelope'] = torch.tensor(clf.predict(np_diffs)).view(1, 1, 256, 256) < 0
    if show >= 1:
        plt.imshow(candidates['EllipticEnvelope'][0, 0].cpu().numpy()), plt.title('EllipticEnvelope'), plt.show()

    # from sklearn.svm import OneClassSVM
    # clf = OneClassSVM()
    # clf.fit(np_diffs)
    # candidates['OneClassSVM'] = torch.tensor(clf.predict(np_diffs)).view(1, 1, 256, 256) < 0
    # if show >= 1:
    #     plt.imshow(candidates['OneClassSVM'][0, 0].cpu().numpy()), plt.title('OneClassSVM'), plt.show()

    # from sklearn.cluster import OPTICS
    # clf = OPTICS()
    # clf.fit(np_diffs)
    # candidates['OPTICS'] = torch.tensor(clf.labels_).view(1, 1, 256, 256) < 0
    # if show >= 1:
    #     plt.imshow(candidates['OPTICS'][0, 0].cpu().numpy()), plt.title('OPTICS'), plt.show()

    return candidates


def nfa_detection_chi2(z, log_nfa_thr, half_win: Union[Tuple, List, int], show=2):
    b, c, h, w = z.shape
    if not isinstance(half_win, (tuple, list)):
        half_win = (half_win, half_win)
    wh, ww = half_win[0], half_win[1]

    log_n_tests = np.log10(h * w)

    # Log-Prob
    z2 = (z ** 2).sum(dim=1, keepdim=True)
    # z2 = (z ** 2)
    z2_tiled = (
        F.pad(z2, tuple([ww, ww, wh, wh]), 'reflect').
        unfold(-2, 2 * wh + 1, 1).
        unfold(-2, 2 * ww + 1, 1)
    )
    z2_max = z2_tiled.amax(dim=(-1, -2))
    # z2_max = z2_tiled.amax(dim=(-1, -2)).amax(dim=1, keepdim=True)
    # c = 1
    log_prob = -(c / 2) * (z2_max / c - 1 - torch.log(z2_max / c)) / np.log(10)

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


def nfa_detection_binomial(z, win=21, show=2):
    half_win = np.max([int(win // 2), 1])
    n_sigmas = 2

    candidates = get_gaussian_fit_candidates(z, show=False, n_sigmas=n_sigmas)['candidates'].sum(dim=1, keepdims=True)
    if show >= 2:
        plt.imshow(candidates[0, 0].cpu().numpy()), plt.show()

    candidates = F.pad(candidates, tuple(4 * [half_win]), 'reflect').detach().cpu()
    candidates_unfold_h = candidates.unfold(-2, 2 * half_win + 1, 1)
    candidates_unfold_hw = candidates_unfold_h.unfold(-2, 2 * half_win + 1, 1).numpy()
    observed_candidates = np.sum(candidates_unfold_hw, axis=(-2, -1))

    n = int((2 * half_win + 1) ** 2)
    log_prob = torch.tensor(st.binom.logsf(observed_candidates, n, 0.997 if n_sigmas == 3 else .95) / np.log(10))

    log_n_tests = np.log10(z.shape[-1] * z.shape[-2])
    log_nfa = log_n_tests + log_prob

    if show >= 2:
        plt.imshow(log_nfa[0, 0].cpu().numpy()), plt.show()

    detection = 1 * (log_nfa < 0)
    if show >= 1:
        plt.imshow(detection[0, 0].cpu().numpy()), plt.show()

    return detection


# def nfa_detection(z, probability_thr=0.99, win=21):
#     tau = st.chi2.ppf(probability_thr, 1)
#     half_win = np.max([int(win // 2), 1])
#
#     n_chann = z.shape[1]
#
#     # Candidates
#     z2 = F.pad(z ** 2, tuple(4 * [half_win]), 'reflect').detach().cpu()
#     z2_unfold_h = z2.unfold(-2, 2 * half_win + 1, 1)
#     z2_unfold_hw = z2_unfold_h.unfold(-2, 2 * half_win + 1, 1).numpy()
#     observed_candidates_k = np.sum(z2_unfold_hw >= tau, axis=(-2, -1))
#
#     # All volume together
#     observed_candidates = np.sum(observed_candidates_k, axis=1, keepdims=True)
#     x = observed_candidates / n_chann
#     n = int((2 * half_win + 1) ** 2)
#
#     log_prob = torch.tensor(st.binom.logsf(x, n, 1 - probability_thr) / np.log(10))
#
#     plt.imshow(observed_candidates[0, 0]), plt.show()
#     plt.imshow(log_prob[0, 0].cpu().numpy()), plt.show()


def get_gaussian_fit_candidates(z, show=True, n_sigmas=3, theoretical_thr=False):

    def gauss_func(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    z_flat = z.cpu().numpy().ravel()
    bin_edges = [np.percentile(z_flat, 0.01), np.percentile(z_flat, 99.99)]
    pdf, bin_edges = np.histogram(z_flat, 1000, bin_edges, density=False)
    bin_width = np.mean(np.diff(bin_edges))
    bins = (bin_edges + bin_width / 2)[:-1]

    # Executing curve_fit on noisy data
    popt, pcov = curve_fit(gauss_func, bins, pdf)
    gain, mean, sigma = popt
    sigma = np.abs(sigma)

    if show:
        estimation = gauss_func(bins, gain, mean, sigma)
        n01 = gauss_func(bins, gain, 0, 1)

        plt.bar(bins, pdf, width=bin_width)
        plt.plot(bins, estimation, c='r')
        plt.plot(bins, n01, c='g')
        plt.show()

    if theoretical_thr:
        candidates = z.abs() > n_sigmas
    else:
        candidates = z.abs() > (mean + n_sigmas * sigma)
    # z_blured = GaussianBlur2d(kernel_size=(3, 3), sigma=(3, 3))(z)
    # candidates = z_blured.abs() > (mean + 2 * sigma)

    candidates = candidates.amax(dim=1, keepdim=True)
    candidates_post = closing(candidates, kernel=torch.ones(3, 3).to(candidates.device))
    candidates_post = opening(candidates_post, kernel=torch.ones(5, 5).to(candidates.device))

    if show:
        plt.imshow(candidates[0, 0].cpu().numpy()), plt.title(f'z > {n_sigmas} sigmas'), plt.show()
        plt.imshow(candidates_post[0, 0].cpu().numpy()), plt.title(f'z > {n_sigmas} sigmas + closing + opening'), plt.show()



    # import cv2
    # c = (255 * candidates[0, 0].numpy()).astype(np.uint8)
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #
    # c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel1)
    # c = cv2.dilate(c, kernel2, iterations=2)
    # c = cv2.morphologyEx(c, cv2.MORPH_OPEN, kernel1)
    # plt.imshow(c), plt.show()


    return {'candidates': candidates, 'candidates_post': candidates_post}


def main():
    # print_likelihood_masks_metrics()
    for category in ["tile"]:  # ALL_CATEGORIES['mvtec']:  # ["tile"]
        # save_z(category)

        test_likelihood_masks(category)
        # save_likelihood_masks_grid(category)
        # save_final_likelihood_masks(category)


if __name__ == '__main__':
    main()

