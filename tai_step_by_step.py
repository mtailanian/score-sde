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

import datasets
from anomaly_detector import AnomalyDetector
from path_utils import get_path
from datasets_anomaly import ALL_CATEGORIES
from utils import load_image_and_label, generate_labels_image, add_blank_row, imshow_tensor, l2
from meta_lpips import MetaLpips
from nfa_tree import compute_nfa_anomaly_score_tree
from dad import DAD
from nfa import nfa_detection_normal, nfa_detection_binomials
from residual_whitening import compute_nfa

sys.setrecursionlimit(100000)


CKPT = 10
SDE = 've'
# SDE = 'subvp'

if SDE == 've':
    from configs.ve import anomaly_256_ncsnpp_continuous as configs
elif SDE == 'subvp':
    from configs.subvp import anomaly_256_ncsnpp_continuous as configs


def save_z(dataset, category):
    # ckpt_path = ckpts[category]
    ckpt_path = str(get_path('training') / 'dad' / SDE / dataset / category / 'checkpoints' / f'checkpoint_{CKPT}.pth')

    out_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = configs.get_config()
    config.data.dataset = dataset
    config.data.category = category

    # n_samples = config.training.batch_size
    # config.training.batch_size = 1
    train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)

    ad = AnomalyDetector(ckpt_path, config)

    ad = DAD(config)
    ad.load_state_dict(torch.load("/home/data/tai/phd/training/tmp/ve/mvtec/carpet/exp_0012/ckpts/TRAIN_6122_loss-1599.85583.ckpt")["state_dict"])

    for batch in tqdm(eval_ds, desc=category):
        # bpd, z, nfe = ad.likelihood_fn(ad.score_model, batch['image'].to(config.device))

        # from skimage.restoration import estimate_sigma
        # sigma_est = np.mean(estimate_sigma(z[0].cpu().numpy(), channel_axis=0))
        # z_nlm = torch.from_numpy(denoise_nl_means(
        #     z[0].cpu().numpy(),
        #     h=1.15 * sigma_est, sigma=sigma_est,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=9, patch_distance=16, channel_axis=0
        # )).unsqueeze(0)
        # imshow_tensor(l2(z_nlm))

        #
        #
        # print(batch['image_path'][0])
        # from likelihood import tai_compute_z
        # tai_compute_z(batch['image'][:1].to(config.device), ad.score_model, ad.sde)
        # bpd, z, nfe = ad.likelihood_fn(ad.score_model, batch['image'][:1].to(config.device))
        #
        # shifted = torch.stack((batch['image'][0][:, 1:, 1:], batch['image'][0][:, :-1, :-1]))
        # shifted = F.pad(shifted, (0, 1, 0, 1), 'constant', 0)
        # bpd, z, nfe = ad.likelihood_fn(ad.score_model, shifted.to(config.device))
        #
        # z_unshifted = torch.stack((z[0, :, :-2, :-2], z[1, :, 1:-1, 1:-1]), dim=0)
        #

        # noisy = batch['image'][:1].repeat(4, 1, 1, 1)
        # noisy += torch.randn_like(noisy) * 5 / 255.
        # bpd, z, nfe = ad.likelihood_fn(ad.score_model, noisy.to(config.device))
        # imshow_tensor(l2(z.mean(dim=0)))




        # noisy1 = batch['image'][0] + 15 / 255. * torch.randn_like(batch['image'][0])
        # noisy2 = batch['image'][0] + 15 / 255. * torch.randn_like(batch['image'][0])
        # bpd, z, nfe = ad.likelihood_fn(ad.score_model, torch.stack((noisy1, noisy2)).to(config.device))
        # imshow_tensor(l2(z[0]))
        # imshow_tensor(l2(z[1]))
        # imshow_tensor(l2(z.mean(dim=0)))

        # Image for diffusion diagram
        # image = torchvision.transforms.ToTensor()(Image.open('/home/dsense/tai/phd/oli.jpg').convert('RGB').resize((128, 128))).unsqueeze(0)
        # t = torch.tensor([0, 0.1, 0.15, 0.2, 0.27, 0.35, 1])
        # z = torch.randn_like(image)
        # mean, std = ad.sde.marginal_prob(image, t)
        # perturbed_data = mean + std[:, None, None, None] * z
        # grid = torchvision.utils.make_grid(perturbed_data, nrow=len(t), normalize=True, scale_each=True, padding=8, pad_value=1)
        # imshow_tensor(grid)
        # torchvision.utils.save_image(grid, '/home/dsense/tai/phd/oli_forward.png')

        #
        #
        #
        #
        bpd, z, nfe = ad.likelihood_fn(ad.score_model, batch['image'][:1].to(config.device))
        # bpd, z, nfe = ad.likelihood_fn(ad.score_model, batch['image'].repeat(n_samples, 1, 1, 1).to(config.device))
        # imshow_tensor((z ** 2).sum(dim=1, keepdim=True).sqrt()[0])
        # imshow_tensor((z ** 2).sum(dim=1, keepdim=True).sqrt()[1])
        # imshow_tensor((z ** 2).sum(dim=1, keepdim=True).sqrt()[2])
        # imshow_tensor((z ** 2).sum(dim=1, keepdim=True).sqrt()[3])
        # imshow_tensor((z ** 2).sum(dim=1, keepdim=True).sqrt().mean(dim=0))
        #
        #
        #
        # from likelihood import tai_compute_z
        # from sde_lib import subVPSDE, VESDE
        # z = tai_compute_z(
        #     batch['image'][:1].to(config.device),
        #     ad.score_model,
        #     # subVPSDE(ad.config.model.beta_min, ad.config.model.beta_max, ad.config.model.num_scales),
        #     VESDE(ad.config.model.beta_min, ad.config.model.beta_max, ad.config.model.num_scales),
        # )
        #
        #
        #
        # from likelihood import get_likelihood_fn
        # from sde_lib import VESDE, subVPSDE
        # if SDE == 've':
        #     sde = VESDE(sigma_min=ad.config.model.sigma_min, sigma_max=ad.config.model.sigma_max, N=ad.config.model.num_scales)
        # elif SDE == 'subvp':
        #     sde = subVPSDE(beta_min=ad.config.model.beta_min, beta_max=ad.config.model.beta_max, N=ad.config.model.num_scales)
        # inverse_scaler = datasets.get_data_inverse_scaler(ad.config)
        # likelihood_fn2 = get_likelihood_fn(sde, inverse_scaler, eps=1e-6, rtol=1e-6, atol=1e-6, method='RK45')#, hutchinson_type="Gaussian")
        # _, z2, _ = likelihood_fn2(ad.score_model, batch['image'][:1].to(config.device))
        # imshow_tensor((z ** 2).sum(dim=1, keepdim=True).sqrt()[0])
        # imshow_tensor((z2 ** 2).sum(dim=1, keepdim=True).sqrt()[0])
        # print
        #
        #
        #

        for i, zi in enumerate(z):
            img_path = Path(batch['image_path'][i])
            defect_type = img_path.parents[0].name
            output_path = out_dir / f"{defect_type}_{img_path.stem}.npy"
            np.save(output_path, zi.unsqueeze(0).cpu().numpy())

            # plt.imshow((zi ** 2).sum(dim=0).sqrt().cpu().numpy()), plt.show()
            # z_nlm = torch.from_numpy(denoise_nl_means(
            #     zi.cpu().numpy() / 256,
            #     h=1., sigma=1,
            #     fast_mode=False, preserve_range=True,
            #     patch_size=5, patch_distance=6, channel_axis=0
            # ))
            # plt.imshow((z_nlm ** 2).sum(dim=0).sqrt().cpu().numpy()), plt.show()


def save_z_nlm(dataset, category, out_size=256, nlm_patch_size=21, show=0):
    # nlm_params = {
    #     'h': 1.,
    #     'sigma': 1,
    #     'fast_mode': True,
    #     'preserve_range': True,
    #     'patch_size': nlm_patch_size,
    #     'patch_distance': 16,
    #     'channel_axis': 0
    # }

    sigma = configs.get_config().model.sigma_max
    nlm_params = {
        'h': 0.8 * sigma,
        'sigma': sigma,
        'fast_mode': False,
        'preserve_range': True,
        'patch_size': 9,
        'patch_distance': 16,
        'channel_axis': 0
    }

    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    out_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / f"z_nlm_p{nlm_params['patch_size']}_d{nlm_params['patch_distance']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(str(out_dir / 'nlm_params.yaml'), 'w') as fid:
        yaml.dump(nlm_params, fid, default_flow_style=False)

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):
        z = torch.from_numpy(np.load(z_path))
        # if SDE == 've':
        #     z = z / 256  # HACK!!

        z_nlm = torch.from_numpy(denoise_nl_means(z[0].numpy(), **nlm_params)).unsqueeze(0)
        np.save(out_dir / z_path.name, z_nlm.numpy())


def save_likelihood_masks_grid(dataset, category, out_size=256):
    out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks" / f"ckpt_{CKPT}"
    out_path.mkdir(parents=True, exist_ok=True)

    resize_transform = torchvision.transforms.Resize(out_size, antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"

    all_images = {
        'original': [],
        'label': [],
        'z': [],
        'z_nlm': [],
        'nfa_binom_1': [],
        'nfa_binom_1_post': [],
        'nfa_binom_2': [],
        'nfa_binom_2_post': [],
        'gauss_3s': [],
        'gauss_3s_post': [],
        'gauss_2s': [],
        'gauss_2s_post': [],
        'dbscan': [],
        'dbscan_post': [],
        'isolation_forest': [],
        'isolation_forest_post': [],
        'elliptic_envelope': [],
        'elliptic_envelope_post': [],
    }
    grid_keys = [k for k in all_images.keys() if k != 'label']
    metric_keys = [k for k in all_images.keys() if k not in ['original', 'label', 'z', 'z_nlm']]

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):

        data = load_image_and_label(z_path, out_size)
        all_images['original'].append(data['image_with_label'] / 255)
        all_images['label'].append(data['label'] / 255)

        z = torch.from_numpy(np.load(z_path)) / 256  # HACK!!

        z_nlm = torch.from_numpy(denoise_nl_means(
            z[0].numpy(),
            h=1., sigma=1,
            fast_mode=False, preserve_range=True,
            patch_size=5, patch_distance=6, channel_axis=0
        )).unsqueeze(0)

        l2 = (z ** 2).sum(dim=1, keepdims=True).sqrt()
        l2 -= l2.min()
        l2 /= l2.quantile(.99)
        all_images['z'].append(resize_transform(l2)[0])

        l2 = (z_nlm ** 2).sum(dim=1, keepdims=True).sqrt()
        l2 -= l2.min()
        l2 /= l2.quantile(.99)
        all_images['z_nlm'].append(resize_transform(l2)[0])

        # %% NFA detections
        result = nfa_detection_binomials(
            z_nlm,
            log_nfa_thr=3,
            n_sigmas_list=[4, 3.5, 3, 2.5, 2, 1.5],
            show=0
        )
        all_images['nfa_binom_1'].append(resize_transform(result['detection'])[0])
        all_images['nfa_binom_1_post'].append(resize_transform(result['detection_post'])[0])

        result = nfa_detection_binomials(
            z_nlm,
            log_nfa_thr=3,
            n_sigmas_list=[4, 3.5, 3, 2.5, 2, 1.5, 1.0],
            show=0
        )
        all_images['nfa_binom_2'].append(resize_transform(result['detection'])[0])
        all_images['nfa_binom_2_post'].append(resize_transform(result['detection_post'])[0])

        # %% Gaussian detections
        result = get_gaussian_fit_candidates(z, z_nlm, n_sigmas=3, show=False, theoretical_thr=True)
        all_images['gauss_3s'].append(resize_transform(result['detection'])[0])
        all_images['gauss_3s_post'].append(resize_transform(result['detection_post'])[0])

        result = get_gaussian_fit_candidates(z, z_nlm, n_sigmas=2, show=False, theoretical_thr=True)
        all_images['gauss_2s'].append(resize_transform(result['detection'])[0])
        all_images['gauss_2s_post'].append(resize_transform(result['detection_post'])[0])

        # %% Clustering detections
        result = clustering_detection(z_nlm, show=0)
        all_images['dbscan'].append(resize_transform(result['dbscan'])[0])
        all_images['dbscan_post'].append(resize_transform(result['dbscan_post'])[0])
        all_images['isolation_forest'].append(resize_transform(result['isolation_forest'])[0])
        all_images['isolation_forest_post'].append(resize_transform(result['isolation_forest_post'])[0])
        all_images['elliptic_envelope'].append(resize_transform(result['elliptic_envelope'])[0])
        all_images['elliptic_envelope_post'].append(resize_transform(result['elliptic_envelope_post'])[0])

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
    out_img = add_blank_row(out_img, [3, 7, 11], 20, out_size, padding=2)

    # plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.show()

    torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")


def save_likelihood_masks_grid_new(dataset, category, out_size=256, show=0):
    if SDE == 've':
        from configs.ve import anomaly_256_ncsnpp_continuous as configs
    elif SDE == 'subvp':
        from configs.subvp import anomaly_256_ncsnpp_continuous as configs
    else:
        raise ValueError(f"Unknown SDE: {SDE}")
    config = configs.get_config()

    uflow_anomaly_dir = get_path('results') / "uflow" / dataset / category / "anomaly_maps"
    uflow_nfa_dir = get_path('results') / "uflow" / dataset / category / "nfa"

    out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks_new" / f"ckpt_{CKPT}"
    out_path.mkdir(parents=True, exist_ok=True)

    resize_transform = torchvision.transforms.Resize(out_size, antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    z_nlm_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z_nlm_p21_d16"

    # nlm_params = yaml.safe_load(open(str(z_nlm_dir / 'nlm_params.yaml'), 'r'))

    all_images = {
        'original': [],
        'label': [],
        'z': [],
        'z_nlm': [],
        # 'gauss_3s': [],
        # 'gauss_2s': [],
        # 'dbscan': [],
        # 'isolation_forest': [],
        # 'elliptic_envelope': [],
        'nfa_w1_thr0': [],
        'nfa_w1_thr2': [],
        'nfa_w3_median_thr0': [],
        'nfa_w3_median_thr2': [],
        'nfa_w3_max_thr0': [],
        'nfa_w3_max_thr2': [],
        # 'otsu': [],
        'uflow': [],
        'uflow_det': [],
    }
    grid_keys = [k for k in all_images.keys() if k != 'label']
    metric_keys = [k for k in all_images.keys() if k not in ['original', 'label', 'z', 'z_nlm', 'uflow']]

    z_paths = natsorted(list(z_dir.glob("*.npy")))  # [::-1][:10]
    for z_path in tqdm(z_paths, desc=category):

        all_images['uflow'].append(resize_transform(torch.from_numpy(np.load(str(uflow_anomaly_dir / z_path.name)))[0]))
        all_images['uflow_det'].append(resize_transform(torch.from_numpy(np.load(str(uflow_nfa_dir / z_path.name)))[0] > 0))

        data = load_image_and_label(z_path, out_size)
        all_images['original'].append(data['image_with_label'] / 255)
        all_images['label'].append(data['label'] / 255)

        z = torch.from_numpy(np.load(z_path))
        # if SDE == 've':
        #     z = z / 256  # HACK!!

        # z_nlm = torch.from_numpy(np.load(str(z_nlm_dir / Path(z_path).name)))
        nlm_params = {
            'h': 0.8 * config.model.sigma_max,
            'sigma': config.model.sigma_max,
            'fast_mode': True,
            'preserve_range': True,
            'patch_size': 5,
            'patch_distance': 16,
            'channel_axis': 0
        }
        z_nlm = torch.from_numpy(denoise_nl_means(z[0].numpy(), **nlm_params)).unsqueeze(0)
        # z_nlm = GaussianBlur2d(kernel_size=(5, 5), sigma=(5, 5))(z_nlm)
        # z_nlm = median_blur(z_nlm, (5, 5))

        l2_norm = l2(z)
        l2_norm -= l2_norm.min()
        l2_norm /= l2_norm.quantile(.99)
        all_images['z'].append(resize_transform(l2_norm)[0])
        if show >= 2:
            imshow_tensor(l2_norm, "z")

        l2_norm = (z_nlm ** 2).sum(dim=1, keepdims=True).sqrt()
        l2_norm -= l2_norm.min()
        l2_norm /= l2_norm.quantile(.99)
        all_images['z_nlm'].append(resize_transform(l2_norm)[0])
        if show >= 2:
            imshow_tensor(l2_norm, "z nlm")

        # %% NFA tree
        # resize2 = torchvision.transforms.Resize(128, antialias=True)   # , interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        # nfa_tree = compute_nfa_anomaly_score_tree([resize2(z_nlm)], target_size=256, sigma=1./5, reduce='min')
        # nfa_tree_detection = nfa_tree > 0
        # imshow_tensor(nfa_tree_detection, "nfa tree")

        # %% Gaussian detections
        # sigma = 1. / nlm_params['patch_size']
        # det_3_sigmas = 1. * (z_nlm.abs().amax(dim=1, keepdim=True) > (3 * sigma))
        # all_images['gauss_3s'].append(resize_transform(det_3_sigmas)[0])
        # if show >= 2:
        #     imshow_tensor(det_3_sigmas, "det_3_sigmas")
        #
        # det_2_sigmas = 1. * (z_nlm.abs().amax(dim=1, keepdim=True) > (2 * sigma))
        # all_images['gauss_2s'].append(resize_transform(det_2_sigmas)[0])
        # if show >= 2:
        #     imshow_tensor(det_2_sigmas, "det_2_sigmas")

        # %% Clustering detections
        # result = clustering_detection(z_nlm, show=0)
        # all_images['dbscan'].append(resize_transform(result['dbscan'])[0])
        # all_images['isolation_forest'].append(resize_transform(result['isolation_forest'])[0])
        # all_images['elliptic_envelope'].append(resize_transform(result['elliptic_envelope'])[0])
        #
        # if show >= 2:
        #     imshow_tensor(result['dbscan'], "dbscan")
        #     imshow_tensor(result['isolation_forest'], "isolation_forest")
        #     imshow_tensor(result['elliptic_envelope'], "elliptic_envelope")

        # %% NFA detections
        # log_nfa_pix = nfa_detection_normal(z_nlm, 0, 0, sigma=1. / nlm_params['patch_size'], show=0)['log_nfa']
        # log_nfa_block3_median = nfa_detection_normal(z_nlm, 0, 1, reduce='median', sigma=1. / nlm_params['patch_size'], show=0)['log_nfa']
        # log_nfa_block3_max = nfa_detection_normal(z_nlm, 0, 1, reduce='max', sigma=1. / nlm_params['patch_size'], show=0)['log_nfa']

        #
        #
        #
        #
        from skimage.restoration import estimate_sigma
        sigma = np.mean(estimate_sigma(z_nlm[0].cpu().numpy(), channel_axis=0))
        # sigma = config.model.sigma_max
        # sigma /= 5
        nfa_detection_normal(z_nlm, 0, 0, sigma, show=2)
        #
        #
        #
        #
        log_nfa_pix = nfa_detection_normal(z_nlm, 0, 0, sigma=1. / 5, show=0)['log_nfa']
        log_nfa_block3_median = nfa_detection_normal(z_nlm, 0, 1, reduce='median', sigma=1. / 5, show=0)['log_nfa']
        log_nfa_block3_max = nfa_detection_normal(z_nlm, 0, 1, reduce='max', sigma=1. / 5, show=0)['log_nfa']

        all_images['nfa_w1_thr0'].append(resize_transform(log_nfa_pix < 0)[0])
        all_images['nfa_w1_thr2'].append(resize_transform(log_nfa_pix < 2)[0])
        all_images['nfa_w3_median_thr0'].append(resize_transform(log_nfa_block3_median < 0)[0])
        all_images['nfa_w3_median_thr2'].append(resize_transform(log_nfa_block3_median < 2)[0])
        all_images['nfa_w3_max_thr0'].append(resize_transform(log_nfa_block3_max < 0)[0])
        all_images['nfa_w3_max_thr2'].append(resize_transform(log_nfa_block3_max < 2)[0])

        if show >= 2:
            # imshow_tensor(log_nfa_pix, "nfa_pix")
            # imshow_tensor(log_nfa_block3_max, "nfa_block3_max")

            # imshow_tensor(log_nfa_pix < 0, "nfa_pix_thr0")
            imshow_tensor(log_nfa_pix < 2, "nfa_pix_thr2")
            # imshow_tensor(log_nfa_block3_median < 0, "nfa_block3_median_thr0")
            imshow_tensor(log_nfa_block3_median < 2, "nfa_block3_median_thr2")
            # imshow_tensor(log_nfa_block3_max < 0, "nfa_block3_max_thr0")
            imshow_tensor(log_nfa_block3_max < 2, "nfa_block3_max_thr2")

        # %% Otsu
        # l22 = (z_nlm ** 2).sum(dim=1, keepdim=True)
        # otsu_thresholds = threshold_multiotsu(l22[0, 0].numpy())
        # otsu_detection = 1. * (l22 > otsu_thresholds.max())
        # all_images['otsu'].append(resize_transform(otsu_detection)[0])
        # if show >= 2:
        #     imshow_tensor(otsu_detection, "otsu_detection")

        print

    # %% Extract metrics
    all_mious = {key: mIoU(thresholds=0.5) for key in metric_keys}
    all_recalls = {key: Recall() for key in metric_keys}
    for k in metric_keys:
        for i in range(len(z_paths)):
            all_mious[k].update(all_images[k][i].float(), all_images['label'][i].float())
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

    # Apply false color to detections
    for key in metric_keys:
        for i in range(len(all_images[key])):
            all_images[key][i] = false_color_img(all_images[key][i][:1] > 0, all_images['label'][i][:1] / 255 > 0)

    # Generate final image
    out_img = torchvision.utils.make_grid(
        torch.cat([all_images[k] for k in grid_keys], dim=0),
        normalize=False, nrow=len(z_paths), padding=2, pad_value=255
    )
    # Add labels
    labels_image = generate_labels_image(list(grid_keys), out_size, padding=2, font_scale=0.8)
    out_img = torch.concat([labels_image, out_img], dim=2)

    # Add blank rows
    out_img = add_blank_row(out_img, [3, 5, 8], 20, out_size, padding=2)

    # plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.show()

    torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")


def save_likelihood_masks_grid_nfa(dataset, category, out_size=256, nlm_patch_size=10):
    out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks_nfa" / f"ckpt_{CKPT}"
    out_path.mkdir(parents=True, exist_ok=True)

    resize_transform = torchvision.transforms.Resize(out_size, antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"

    nfa_thresholds = [0, -2, -10, 2, 5]
    nfa_half_windows = [0, 1]

    all_images = {
        'original': [],
        'label': [],
        'z': [],
        'z_nlm': [],
    }
    for nfa_thr in nfa_thresholds:
        for nfa_hw in nfa_half_windows:
            key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
            all_images[key] = []

    grid_keys = [k for k in all_images.keys() if k != 'label']
    metric_keys = [k for k in all_images.keys() if k not in ['original', 'label', 'z', 'z_nlm']]

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):

        data = load_image_and_label(z_path, out_size)
        all_images['original'].append(data['image_with_label'] / 255)
        all_images['label'].append(data['label'] / 255)

        z = torch.from_numpy(np.load(z_path))
        if SDE == 've':
            z = z / 256  # HACK!!

        z_nlm = torch.from_numpy(denoise_nl_means(
            z[0].numpy(),
            h=1., sigma=1,
            fast_mode=False, preserve_range=True,
            patch_size=nlm_patch_size, patch_distance=6, channel_axis=0
        )).unsqueeze(0)

        l2 = (z ** 2).sum(dim=1, keepdims=True).sqrt()
        l2 -= l2.min()
        l2 /= l2.quantile(.99)
        all_images['z'].append(resize_transform(l2)[0])

        l2 = (z_nlm ** 2).sum(dim=1, keepdims=True).sqrt()
        l2 -= l2.min()
        l2 /= l2.quantile(.99)
        all_images['z_nlm'].append(resize_transform(l2)[0])

        for nfa_thr in nfa_thresholds:
            for nfa_hw in nfa_half_windows:
                key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
                detection = nfa_detection_normal(z_nlm, nfa_thr, nfa_hw, sigma=1. / nlm_patch_size, show=0)['detection'][0]
                all_images[key].append(resize_transform(detection))

    # Extract metrics
    # ------------------------------------------------------------------------------------------------------------------
    all_mious = {key: mIoU(thresholds=0.5) for key in metric_keys}
    all_recalls = {key: Recall() for key in metric_keys}
    for k in metric_keys:
        for i in range(len(z_paths)):
            all_mious[k].update(all_images[k][i].float(), all_images['label'][i].float())
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

    # Generate final grid image
    # ------------------------------------------------------------------------------------------------------------------
    for k in grid_keys:
        all_images[k] = torch.stack(all_images[k])
        if all_images[k].shape[1] == 1:
            all_images[k] = all_images[k].repeat(1, 3, 1, 1)

    # Apply false color to detections
    for nfa_thr in nfa_thresholds:
        for nfa_hw in nfa_half_windows:
            key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
            for i in range(len(all_images[key])):
                all_images[key][i] = false_color_img(all_images[key][i][:1], all_images['label'][i][:1] / 255)

    # Generate final image
    out_img = torchvision.utils.make_grid(
        torch.cat([all_images[k] for k in grid_keys], dim=0),
        normalize=False, nrow=len(z_paths), padding=2, pad_value=1.
    )

    # Add labels
    labels_image = generate_labels_image(list(grid_keys), out_size, padding=2, font_scale=0.8)
    out_img = torch.concat([labels_image, out_img], dim=2)

    # Add blank rows
    out_img = add_blank_row(out_img, [3, 4, 6], 20, out_size, padding=2)

    # plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.show()

    torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")


def uflow_comparison(dataset, category, out_size=256, nlm_patch_size=5, show=0):
    out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "uflow_comparison"
    out_path.mkdir(parents=True, exist_ok=True)

    resize_transform = torchvision.transforms.Resize(out_size, antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    uflow_anomaly_dir = get_path('results') / "uflow" / dataset / category / "anomaly_maps"
    uflow_nfa_dir = get_path('results') / "uflow" / dataset / category / "nfa"

    nfa_thresholds = [0, 2]
    # nfa_half_windows = [0, 1]

    all_images = {
        'original': [],
        'label': [],
        'z': [],
        'z_nlm': [],
        "nfa_thr2": [],
        "nfa_post": [],
        "nfa_thr0": [],
        'uflow_det': [],
        'uflow': [],
    }
    # for nfa_thr in nfa_thresholds:
    #     for nfa_hw in nfa_half_windows:
    #         key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
    #         all_images[key] = []

    grid_keys = [k for k in all_images.keys() if k != 'label']
    metric_keys = [k for k in all_images.keys() if k not in ['original', 'label', 'z', 'z_nlm', 'uflow']]

    z_paths = natsorted(list(z_dir.glob("*.npy")))  # [::-1][:10]
    for z_path in tqdm(z_paths, desc=category):
        # if "good" in str(z_path):
        #     continue

        data = load_image_and_label(z_path, out_size)
        all_images['original'].append(data['image_with_label'] / 255)
        all_images['label'].append(data['label'] / 255)

        z = torch.from_numpy(np.load(z_path))

        z_nlm = torch.from_numpy(np.load(z_path.parent.parent / "z_nlm_p9_d16" / z_path.name))

        # if SDE == 've':
        #     z = z / 256  # HACK!!

        # z_nlm = torch.from_numpy(denoise_nl_means(
        #     z[0].numpy(),
        #     h=1., sigma=1,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=nlm_patch_size,
        #     # patch_distance=6,
        #     patch_distance=16,
        #     # patch_distance=36,
        #     channel_axis=0
        # )).unsqueeze(0)

        z_l2 = l2(z)
        z_l2 -= z_l2.min()
        z_l2 /= z_l2.quantile(.99)
        z_l2 = torch.clip(z_l2, 0, 1)
        all_images['z'].append(resize_transform(z_l2)[0])
        if show >= 2:
            imshow_tensor(z_l2)

        z_nlm_l2 = l2(z_nlm)
        z_nlm_l2 -= z_nlm_l2.min()
        z_nlm_l2 /= z_nlm_l2.quantile(.999)
        z_nlm_l2 = torch.clip(z_nlm_l2, 0, 1)
        all_images['z_nlm'].append(resize_transform(z_nlm_l2)[0])
        if show >= 2:
            imshow_tensor(z_nlm_l2)

        for nfa_thr in nfa_thresholds:
            key = f"nfa_thr{nfa_thr}"
            nfa = torch.from_numpy(compute_nfa(z_nlm[0].permute(1, 2, 0).cpu().numpy())).permute(2, 0, 1)
            detection = nfa < nfa_thr
            all_images[key].append(resize_transform(detection))

            # for nfa_hw in nfa_half_windows:
            #     key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
            #     detection = nfa_detection_normal(z_nlm, nfa_thr, nfa_hw, sigma=1. / nlm_patch_size, show=0)['detection'][0]
            #     all_images[key].append(resize_transform(detection))

        # post = torch.from_numpy(denoise_nl_means(
        #     all_images["nfa_thr0_win1"][-1][0].numpy(),
        #     h=1., sigma=1,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=nlm_patch_size, patch_distance=6
        # )).unsqueeze(0) > 0.5
        # all_images['nfa_post'].append(resize_transform(post))

        # post = GaussianBlur2d(kernel_size=(5, 5), sigma=(5, 5))(all_images["nfa_thr0_win3"][-1].unsqueeze(0).float())
        device = 'cpu'
        k3 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32).to(device)
        k5 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32).to(device)
        # post = F.pad(all_images["nfa_thr2"][-1], (2, 2, 2, 2), mode='constant', value=0)
        # post = dilation(post.unsqueeze(0), k5)[0]
        # post = post[:, 2:-2, 2:-2]
        post = torch.from_numpy(median_filter(all_images["nfa_thr2"][-1].cpu().numpy(), size=3)).unsqueeze(0)
        post = opening(1 * post, kernel=k3)
        post = F.pad(post, (1, 1, 1, 1), mode='constant', value=0)
        post = dilation(post, kernel=k5)
        post = post[0, :, 1:-1, 1:-1]

        all_images['nfa_post'].append(resize_transform(post > 0.5))
        # post = dilation(post, ksmall)
        # post = GaussianBlur2d(kernel_size=(5, 5), sigma=(5, 5))(post)[0]
        # all_images['nfa_post'].append(resize_transform(post > 0.5))
        if show >= 2:
            imshow_tensor(all_images["nfa_thr0_win3"][-1])
            imshow_tensor(post)
            # imshow_tensor(post > 0.5)

        # UFLOW
        all_images['uflow'].append(resize_transform(torch.from_numpy(np.load(str(uflow_anomaly_dir / z_path.name)))[0]))
        all_images['uflow_det'].append(resize_transform(torch.from_numpy(np.load(str(uflow_nfa_dir / z_path.name)))[0] > 0))

    # Extract metrics
    # ------------------------------------------------------------------------------------------------------------------
    all_mious = {key: mIoU(thresholds=0.5) for key in metric_keys}
    all_recalls = {key: Recall() for key in metric_keys}
    for k in metric_keys:
        for i in range(len(all_images[k])):
            all_mious[k].update(all_images[k][i].float(), all_images['label'][i].float())
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

    # Generate final grid image
    # ------------------------------------------------------------------------------------------------------------------
    for k in grid_keys:
        all_images[k] = torch.stack(all_images[k])
        if all_images[k].shape[1] == 1:
            all_images[k] = all_images[k].repeat(1, 3, 1, 1)

    # Apply false color to detections
    for nfa_thr in nfa_thresholds:
        key = f"nfa_thr{nfa_thr}"
        for i in range(len(all_images[key])):
            all_images[key][i] = false_color_img(all_images[key][i][:1], all_images['label'][i][:1] / 255)
        # for nfa_hw in nfa_half_windows:
        #     key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
        #     for i in range(len(all_images[key])):
        #         all_images[key][i] = false_color_img(all_images[key][i][:1], all_images['label'][i][:1] / 255)
    for key in ['nfa_post', 'uflow_det']:
        for i in range(len(all_images[key])):
            all_images[key][i] = false_color_img(all_images[key][i][:1], all_images['label'][i][:1] / 255)

    # Generate final image
    out_img = torchvision.utils.make_grid(
        torch.cat([all_images[k] for k in grid_keys], dim=0),
        normalize=False, nrow=len(z_paths), padding=2, pad_value=1.
    )

    # Add labels
    labels_image = generate_labels_image(list(grid_keys), out_size, padding=2, font_scale=0.8) / 255.
    out_img = torch.concat([labels_image, out_img], dim=2)

    # Add blank rows
    out_img = add_blank_row(out_img, [3, 6], 20, out_size, padding=2)

    # plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.show()

    torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")


def print_likelihood_masks_metrics(dataset):
    # out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks" / f"ckpt_{CKPT}"
    # out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks_nfa" / f"ckpt_{CKPT}"
    # out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks_nfa_median" / f"ckpt_{CKPT}"
    out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks_new" / f"ckpt_{CKPT}"

    metrics = {}
    for category in ALL_CATEGORIES[dataset]:
        with open(str(out_path / f"metrics_{category}.yaml"), 'r') as yaml_file:
            metrics[category] = yaml.safe_load(yaml_file)

    sep = " "
    methods = metrics[ALL_CATEGORIES[dataset][0]].keys()
    print(f"{'Category'}{sep}{(sep + sep).join([k for k in methods])}")
    print(f"-{sep}{sep.join([f'mIoU{sep}Recall' for k in methods])}")
    for category in ALL_CATEGORIES[dataset]:
        print(f"{category}", end=sep)
        for method in methods:
            print(f"{100 * metrics[category][method]['mIoU']:.2f}{sep}{100 * metrics[category][method]['Recall']:.2f}", end=sep)
        print()


def save_final_likelihood_masks(dataset, category):
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    mask_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "masks"
    for method in ['nfa_binom_1', 'nfa_binom_2', 'gauss_3s', 'gauss_2s', 'isolation_forest', 'elliptic_envelope']:
        (mask_dir / method).mkdir(parents=True, exist_ok=True)

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):

        z = torch.from_numpy(np.load(z_path)) / 256  # HACK!!

        z_nlm = torch.from_numpy(denoise_nl_means(
            z[0].numpy(),
            h=1., sigma=1,
            fast_mode=False, preserve_range=True,
            patch_size=5, patch_distance=6, channel_axis=0
        )).unsqueeze(0)

        # MASKS
        nfa_binom_1 = nfa_detection_binomials(
            z_nlm,
            log_nfa_thr=3,
            n_sigmas_list=[4, 3.5, 3, 2.5, 2, 1.5],
            show=0
        )['detection']
        np.save(mask_dir / 'nfa_binom_1' / z_path.name, nfa_binom_1.cpu().numpy())

        nfa_binom_2 = nfa_detection_binomials(
            z_nlm,
            log_nfa_thr=3,
            n_sigmas_list=[4, 3.5, 3, 2.5, 2, 1.5, 1.0],
            show=0
        )['detection']
        np.save(mask_dir / 'nfa_binom_2' / z_path.name, nfa_binom_2.cpu().numpy())

        gauss_3s = get_gaussian_fit_candidates(z, z_nlm, n_sigmas=3, show=False, theoretical_thr=True)['detection']
        np.save(mask_dir / 'gauss_3s' / z_path.name, gauss_3s.cpu().numpy())

        gauss_2s = get_gaussian_fit_candidates(z, z_nlm, n_sigmas=2, show=False, theoretical_thr=True)['detection']
        np.save(mask_dir / 'gauss_2s' / z_path.name, gauss_2s.cpu().numpy())

        result = clustering_detection(z_nlm, show=0)
        isolation_forest = result['isolation_forest']
        np.save(mask_dir / 'isolation_forest' / z_path.name, isolation_forest.cpu().numpy())
        elliptic_envelope = result['elliptic_envelope']
        np.save(mask_dir / 'elliptic_envelope' / z_path.name, elliptic_envelope.cpu().numpy())


def save_final_likelihood_masks_nfa(dataset, category, nlm_patch_size=5):
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    mask_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "masks"

    nfa_thresholds = [-10]
    nfa_half_windows = [0]
    for nfa_thr in nfa_thresholds:
        for nfa_hw in nfa_half_windows:
            key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
            (mask_dir / key).mkdir(parents=True, exist_ok=True)

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):

        z = torch.from_numpy(np.load(z_path)) / 256  # HACK!!

        z_nlm = torch.from_numpy(denoise_nl_means(
            z[0].numpy(),
            h=1., sigma=1,
            fast_mode=False, preserve_range=True,
            patch_size=nlm_patch_size, patch_distance=6, channel_axis=0
        )).unsqueeze(0)

        for nfa_thr in nfa_thresholds:
            for nfa_hw in nfa_half_windows:
                key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"
                mask = nfa_detection_normal(
                    z_nlm, half_win=nfa_hw, log_nfa_thr=nfa_thr, sigma=1. / nlm_patch_size, show=0
                )['detection'][0]
                np.save(mask_dir / key / z_path.name, mask.cpu().numpy())


def refine_postprocessing(dataset, category):
    device = "cpu"
    k3 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32).to(device)
    k5 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32).to(device)
    k7 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).type(torch.float32).to(device)
    k21 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).type(torch.float32).to(device)

    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    z_paths = natsorted(list(z_dir.glob("*.npy")))
    # Exclude good images to avoid computing individual metrics for them
    # z_paths = [z_path for z_path in z_paths if 'good' not in z_path.stem]
    for method in ['isolation_forest']:  # ['nfa_binom_1', 'nfa_binom_2', 'gauss_3s', 'gauss_2s', 'isolation_forest', 'elliptic_envelope'][::-1]:
        recall = Recall()
        individual_recalls = []
        show_images = []
        for z_path in tqdm(z_paths, desc=f"{category}> {method}"):
            data = load_image_and_label(z_path, 256)
            label = data['label'].unsqueeze(0) / 255
            likelihood_mask = torch.from_numpy(1 * np.load(z_dir.with_name('masks') / method / z_path.name))

            # Gauss 2
            # detection_post = opening(1 * likelihood_mask, kernel=k3)
            # detection_post = dilation(detection_post, kernel=k5)
            # detection_post = closing(detection_post, kernel=k5)
            # detection_post = dilation(detection_post, kernel=k5)

            # Isolation Forest
            detection_post = opening(1 * likelihood_mask, kernel=k3)
            detection_post = dilation(detection_post, kernel=k21)

            det_gt = ((detection_post > 0.5).float().ravel(), (label > 0.5).float().ravel())
            recall.update(det_gt)
            recall_i = Recall()
            recall_i.update(det_gt)
            individual_recalls.append(recall_i.compute().item())

            show_images.append(data['image_with_label'] / 255)
            show_images.append(detection_post[0].repeat(3, 1, 1))

        print(f"{method}: {100 * recall.compute().item():.2f}")
        individual_recalls = np.array(individual_recalls)
        worst_indexes = np.argsort(individual_recalls)
        # _ = [print(f"{np.array(z_paths)[wi].stem}\t{individual_recalls[wi] * 100:.2f}") for wi in worst_indexes[:10]]

        out_img = torchvision.utils.make_grid(show_images, normalize=False, nrow=14, padding=2, pad_value=1)
        plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.xticks([]), plt.yticks([]), plt.axis('equal'), plt.axis('tight'), plt.tight_layout(), plt.show()

        out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks" / f"ckpt_{CKPT}" / method / "open3-dil21"
        out_path.mkdir(parents=True, exist_ok=True)
        # torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")
        print


def refine_postprocessing_nfa(dataset, category):
    device = "cpu"
    ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32).to(device)
    ksmall2 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32).to(device)
    kbig1 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).type(torch.float32).to(device)
    kbig2 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).type(torch.float32).to(device)

    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    z_paths = natsorted(list(z_dir.glob("*.npy")))
    # Exclude good images to avoid computing individual metrics for them
    # z_paths = [z_path for z_path in z_paths if 'good' not in z_path.stem]

    nfa_thresholds = [-10]
    nfa_half_windows = [0]
    for nfa_thr in nfa_thresholds:
        for nfa_hw in nfa_half_windows:
            key = f"nfa_thr{nfa_thr}_win{2 * nfa_hw + 1}"

            recall = Recall()
            individual_recalls = []
            show_images = []
            for z_path in tqdm(z_paths, desc=f"{category}> {key}"):
                data = load_image_and_label(z_path, 256)
                label = data['label'].unsqueeze(0) / 255
                likelihood_mask = torch.from_numpy(1 * np.load(z_dir.with_name('masks') / key / z_path.name)).unsqueeze(0)

                # detection_post = likelihood_mask
                # if nfa_hw == 0:
                #     detection_post = opening(1 * likelihood_mask, kernel=ksmall)
                # else:
                #     detection_post = likelihood_mask
                # # detection_post = dilation(detection_post, kernel=kbig)
                # detection_post = closing(detection_post, kernel=kbig2)
                # detection_post = dilation(detection_post, kernel=kbig1)

                # New for NFA with threshold -10
                detection_post = closing(likelihood_mask, kernel=kbig1)
                detection_post = opening(detection_post, kernel=ksmall)
                detection_post = dilation(detection_post, kernel=kbig1)

                det_gt = ((detection_post > 0.5).float().ravel(), (label > 0.5).float().ravel())
                recall.update(det_gt)
                recall_i = Recall()
                recall_i.update(det_gt)
                individual_recalls.append(recall_i.compute().item())

                show_images.append(data['image_with_label'] / 255)
                # show_images.append(detection_post[0].repeat(3, 1, 1))
                show_images.append(false_color_img(detection_post[0], label[0]))

            print(f"{key}: {100 * recall.compute().item():.2f}")
            individual_recalls = np.array(individual_recalls)
            worst_indexes = np.argsort(individual_recalls)
            # _ = [print(f"{np.array(z_paths)[wi].stem}\t{individual_recalls[wi] * 100:.2f}") for wi in worst_indexes[:10]]

            out_img = torchvision.utils.make_grid(show_images, normalize=False, nrow=14, padding=2, pad_value=1)
            plt.imshow(out_img.permute(1, 2, 0).numpy()), plt.xticks([]), plt.yticks([]), plt.axis('equal'), plt.axis('tight'), plt.tight_layout(), plt.show()

            out_path = get_path('results') / "dad" / SDE / "debug" / dataset / "likelihood_masks_nfa" / f"ckpt_{CKPT}" / key / "morphology"
            out_path.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(out_img, out_path / f"{category}.pdf")
            print


def inpainting(dataset, category, n_samples=1):
    # mask_method = 'IsolationForest'
    # mask_method = 'nfa_thr-10_win1'
    mask_method = 'nfa_thibaud'

    masks_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "masks" / mask_method
    out_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "inpainting" / mask_method
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(get_path('training') / 'dad' / SDE / dataset / category / 'checkpoints' / f'checkpoint_{CKPT}.pth')
    ad = AnomalyDetector(ckpt_path)

    if mask_method == 'IsolationForest':
        ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32).to(ad.config.device)
        kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).type(torch.float32).to(ad.config.device)
    elif mask_method == 'nfa_thr-10_win1':
        ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32).to(ad.config.device)
        kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).type(torch.float32).to(ad.config.device)
    elif mask_method == 'nfa_thibaud':
        pass
    else:
        raise NotImplementedError

    batch_size = 32
    paths = list(natsorted(masks_dir.glob("*.npy")))
    for i in tqdm(range(0, len(paths), batch_size), desc=category):

        if (out_dir / paths[i].name).exists():
            continue

        batch_paths = paths[i:i + batch_size]
        batch_images = []
        batch_masks = []
        for mask_path in batch_paths:
            defect_type = "_".join(mask_path.stem.split("_")[:-1])
            name = Path(mask_path.stem).stem.split("_")[-1]
            img_path = get_path('data') / dataset / category / "test" / defect_type / f"{name}.png"
            image = ad.transform(Image.open(str(img_path)).convert('RGB')).unsqueeze(0).to(ad.config.device)
            mask = torch.from_numpy(np.load(mask_path)).to(ad.config.device)
            mask = mask.unsqueeze(0) if len(mask.shape) == 3 else mask
            if mask_method == 'IsolationForest':
                mask = opening(1 * mask, kernel=ksmall)
                mask = dilation(mask, kernel=kbig)
            elif mask_method == 'nfa_thr-10_win1':
                mask = closing(mask, kernel=kbig)
                mask = opening(mask, kernel=ksmall)
                mask = dilation(mask, kernel=kbig)
            elif mask_method == 'nfa_thibaud':
                mask = mask  # already saved with morphology post-processing
            batch_images.append(image)
            batch_masks.append(mask)
        batch_images = torch.cat(batch_images, dim=0)
        batch_masks = torch.cat(batch_masks, dim=0)

        inpainted = ad.inpainter(ad.score_model, ad.scaler(batch_images), 1 - batch_masks)

        for j, inpainted_j in enumerate(inpainted):
            np.save(str(out_dir / batch_paths[j].name), inpainted_j.cpu().numpy())

        # inpainted_grid = torchvision.utils.make_grid(torch.cat([batch_images, batch_masks.repeat(1, 3, 1, 1), inpainted], dim=0), normalize=False, nrow=batch_size, padding=2, pad_value=1)
        # imshow_tensor(inpainted_grid)

    # for mask_path in tqdm(paths, desc=category):
    #     if (out_dir / mask_path.name).exists():
    #         continue
    #
    #     defect_type = "_".join(mask_path.stem.split("_")[:-1])
    #     name = Path(mask_path.stem).stem.split("_")[-1]
    #     img_path = get_path('data') / dataset / category / "test" / defect_type / f"{name}.png"
    #     image = ad.transform(Image.open(str(img_path)).convert('RGB')).unsqueeze(0).to(ad.config.device)
    #     image = repeat(image, "1 ... -> n ...", n=n_samples)
    #
    #     mask = torch.from_numpy(np.load(mask_path)).to(ad.config.device)
    #     if mask_method == 'IsolationForest':
    #         mask = opening(1 * mask, kernel=ksmall)
    #         mask = dilation(mask, kernel=kbig)
    #     elif mask_method == 'nfa_thr-10_win1':
    #         mask = closing(mask, kernel=ksmall)
    #         mask = opening(mask, kernel=ksmall)
    #         mask = dilation(mask, kernel=kbig)
    #     mask = repeat(mask, "1 ... -> n ...", n=n_samples)
    #
    #     inpainted = ad.inpainter(ad.score_model, ad.scaler(image), 1 - mask)
    #
    #     np.save(str(out_dir / mask_path.name), inpainted.cpu().numpy())
    #
    #     # imshow_tensor(image)
    #     # imshow_tensor(mask)
    #     # inpainted_grid = torchvision.utils.make_grid(inpainted, normalize=False, nrow=4, padding=2, pad_value=1)
    #     # imshow_tensor(inpainted_grid)


def compute_anomaly_maps(dataset, category):
    # mask_method = "isolation_forest"
    # mask_method = "nfa_thr-10_win1"
    mask_method = "nfa_thibaud"

    inpainting_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "inpainting" / mask_method
    likelihood_mask_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "masks" / mask_method

    out_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "anomaly_map" / mask_method / "lpips"
    out_grid_dir = get_path('results') / "dad" / SDE / "debug" / dataset / "anomaly_maps" / f"ckpt_{CKPT}" / mask_method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_grid_dir.mkdir(parents=True, exist_ok=True)

    if mask_method == 'isolation_forest':
        ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32)
        kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).type(torch.float32)
    elif mask_method == 'nfa_thr-10_win1':
        ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32)
        kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).type(torch.float32)
    elif mask_method == 'nfa_thibaud':
        pass
    else:
        raise NotImplementedError

    lpips = MetaLpips(verbose=False)
    auroc = ROC_AUC()
    all_images, all_labels, all_masks, all_masks_dilation, all_inpainteds, all_maps = [], [], [], [], [], []
    for inpainted_path in tqdm(list(natsorted(inpainting_dir.glob("*.npy"))), desc=category):

        inpainted = torch.from_numpy(np.load(inpainted_path))
        inpainted = inpainted.unsqueeze(0) if len(inpainted.shape) == 3 else inpainted

        data = load_image_and_label(inpainted_path, 256, category_position_in_path=-5)
        image = data['image'].unsqueeze(0) / 255
        label = data['label'].unsqueeze(0) > 127.5

        anomaly_maps = lpips(inpainted, image)
        anomaly_map = torch.from_numpy(anomaly_maps['LPIPS'])

        np.save(out_dir / inpainted_path.name, anomaly_map)

        auroc.update((anomaly_map.ravel(), label.ravel()))

        all_images.append(data['image'] / 255)
        all_labels.append(data['label'] / 255)
        all_maps.append(anomaly_map.unsqueeze(0))
        all_inpainteds.append(inpainted[0])
        lik_mask = torch.from_numpy(np.load(likelihood_mask_dir / inpainted_path.name))[0]
        lik_mask = lik_mask.unsqueeze(0) if len(lik_mask.shape) == 2 else lik_mask
        if mask_method == 'isolation_forest':
            lik_mask_dilated = opening(1 * lik_mask.unsqueeze(0), kernel=ksmall)
            lik_mask_dilated = dilation(lik_mask_dilated, kernel=kbig)
        elif mask_method == 'nfa_thr-10_win1':
            lik_mask_dilated = closing(lik_mask.unsqueeze(0), kernel=kbig)
            lik_mask_dilated = opening(lik_mask_dilated, kernel=ksmall)
            lik_mask_dilated = dilation(lik_mask_dilated, kernel=kbig)
        elif mask_method == 'nfa_thibaud':
            lik_mask_dilated = lik_mask.unsqueeze(0)
        all_masks.append(lik_mask)
        all_masks_dilation.append(lik_mask_dilated[0])

        # for metric_name in ['LPIPS', 'META-LPIPS', 'L1']:
        #     plt.imshow(anomaly_maps[metric_name]), plt.title(metric_name), plt.show()

    print(f"{category}: {auroc.compute().item() * 100:.2f}")

    all_maps = np.concatenate(all_maps, axis=0)
    all_maps -= all_maps.min()
    all_maps /= np.percentile(all_maps, 99)
    all_maps = np.clip(all_maps, 0, 1)
    all_maps = torch.from_numpy(all_maps).unsqueeze(1).repeat(1, 3, 1, 1)
    out_img = torchvision.utils.make_grid(torch.cat([torch.stack(all_images), torch.stack(all_masks).repeat(1, 3, 1, 1), torch.stack(all_masks_dilation).repeat(1, 3, 1, 1), torch.stack(all_inpainteds), all_maps, torch.stack(all_labels).repeat(1, 3, 1, 1)]), normalize=False, nrow=len(all_images), padding=2, pad_value=1)
    labels_image = generate_labels_image(['Image', 'Likelihood mask', 'Dilation', 'Inpainted', 'Anomaly Map', 'Label'], 256, padding=2, font_scale=0.8)
    out_img = torch.cat([labels_image, out_img], dim=2)
    torchvision.utils.save_image(out_img, out_grid_dir / f"{category}.pdf")


def test_likelihood_masks(dataset, category):
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"

    z_paths = natsorted(list(z_dir.glob("*.npy")))[::1]
    # z_paths = [zp for zp in z_paths if 'thread' in str(zp) and '5' in str(zp)]
    for z_path in tqdm(z_paths, desc=category):
        data = load_image_and_label(z_path, 256)
        # imshow_tensor(data['image'], "Original")
        imshow_tensor(data['image_with_label'], "With label")

        label = data['label'].unsqueeze(0) / 255

        z = torch.from_numpy(np.load(z_path))
        z_nlm = torch.from_numpy(np.load(z_path.parent.parent / "z_nlm_p9_d16" / z_path.name))

        # if SDE == "ve":
        #     z = z / 256  # HACK!!

        # patch_size = 5
        # z_denoise = torch.from_numpy(denoise_nl_means(
        #     z[0].numpy(),
        #     h=1., sigma=1,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=patch_size, patch_distance=6, channel_axis=0
        # )).unsqueeze(0)
        # z_denoise = torch.from_numpy(denoise_nl_means(
        #     z[0].numpy(),
        #     h=.8, sigma=1,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=patch_size, patch_distance=6, channel_axis=0
        # )).unsqueeze(0)

        # imshow_tensor(l2(z), 'z: L2')
        imshow_tensor(l2(z_nlm), 'z denoise: L2')

        # result = nfa_detection_binomials(
        #     z_denoise,
        #     log_nfa_thr=3,
        #     n_sigmas_list=[4, 3.5, 3, 2.5, 2, 1.5],
        #     show=1
        # )
        # nfa_detection_binomial_1, nfa_detection_binomial_1_post = result['detection'], result['detection_post']
        #
        # result = nfa_detection_binomials(
        #     z_denoise,
        #     log_nfa_thr=3,
        #     n_sigmas_list=[4, 3.5, 3, 2.5, 2, 1.5, 1.0],
        #     show=1
        # )
        # nfa_detection_binomial_2, nfa_detection_binomial_2_post = result['detection'], result['detection_post']
        #
        # result = get_gaussian_fit_candidates(z, z_denoise, n_sigmas=3, show=True, theoretical_thr=True)
        # gauss_fit_3_sigmas, gauss_fit_3_sigmas_post = result['detection'], result['detection_post']
        #
        # result = get_gaussian_fit_candidates(z, z_denoise, n_sigmas=2, show=True, theoretical_thr=True)
        # gauss_fit_2_sigmas, gauss_fit_2_sigmas_post = result['detection'], result['detection_post']
        #
        # result = clustering_detection(z_denoise, show=1)
        # dbscan, dbscan_post = result['dbscan'], result['dbscan_post']
        # isolation_forest, isolation_forest_post = result['isolation_forest'], result['isolation_forest_post']
        # elliptic_envelope, elliptic_envelope_post = result['elliptic_envelope'], result['elliptic_envelope_post']

        # NFA Normal Distribution with NLM average
        # Check that the denoise z has sigma = 1 / sqrt(number of averaged pixels)
        # bins, pdf, bin_width = get_histogram(z)
        # popt, _ = curve_fit(gauss_func, bins, pdf)
        # gain, mean, sigma = popt
        #
        # bins_denoise, pdf_denoise, bin_width_denoise = get_histogram(z_denoise)
        # popt_denoise, _ = curve_fit(gauss_func, bins_denoise, pdf_denoise)
        # gain_denoise, mean_denoise, sigma_denoise = popt_denoise
        #
        # plt.bar(bins, pdf, width=bin_width, alpha=0.5)
        # plt.bar(bins_denoise, pdf_denoise, width=bin_width_denoise, alpha=0.5)
        # plt.plot(bins, gauss_func(bins, gain, 0, 1), c='r')
        # plt.plot(bins, gauss_func(bins, gain_denoise, 0, 1. / 5), c='g')
        # plt.show()
        # plt.xlim(-4, 4)

        # nfa_detection_normal(z_denoise, half_win=0, log_nfa_thr=0, sigma=1. / 5, show=1)['detection']
        # nfa_detection_normal(z_denoise, half_win=0, log_nfa_thr=-2, sigma=1. / 5, show=1)['detection']
        # nfa_detection_normal(z_denoise, half_win=0, log_nfa_thr=-4, sigma=1. / 5, show=1)['detection']
        # nfa_detection_normal(z_denoise, half_win=0, log_nfa_thr=-8, sigma=1. / 5, show=1)['detection']
        # detection = nfa_detection_normal(z_denoise, half_win=0, log_nfa_thr=-10, sigma=1. / 5, show=1)['detection']

        # detection = nfa_detection_normal(z_denoise, half_win=0, log_nfa_thr=0, sigma=1. / patch_size, show=0)['detection']

        nfa = torch.from_numpy(compute_nfa(z_nlm[0].permute(1, 2, 0).cpu().numpy())).permute(2, 0, 1).unsqueeze(0)
        detection = nfa < 3
        imshow_tensor(detection)
        # from scipy.ndimage import median_filter
        # out = torch.from_numpy(median_filter(detection, size=3))
        # imshow_tensor(out)

        kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).type(torch.float32)
        ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32)
        detection_post = closing(detection, kernel=ksmall)
        # detection_post = opening(detection_post, kernel=ksmall)
        detection_post = torch.from_numpy(median_filter(detection_post, size=3))
        detection_post = dilation(detection_post, kernel=kbig)
        # imshow_tensor(detection_post, "Post")
        fci = false_color_img(detection_post[0], load_image_and_label(z_path, 256)['label'] / 255)
        plt.imshow(fci.permute(1, 2, 0).cpu().numpy()), plt.title('False color'), plt.show()
        print()

        # nfa_detection_normal(z_denoise, half_win=1, log_nfa_thr=2, sigma=1. / 5, show=2)['detection']
        # nfa_detection_normal(z_denoise, half_win=1, log_nfa_thr=1, sigma=1. / 5, show=1)['detection']
        # nfa_detection_normal(z_denoise, half_win=1, log_nfa_thr=0, sigma=1. / 5, show=1)['detection']
        #
        # nfa_detection_normal(z_denoise, half_win=2, log_nfa_thr=2, sigma=1. / 5, show=2)['detection']
        # nfa_detection_normal(z_denoise, half_win=2, log_nfa_thr=1, sigma=1. / 5, show=1)['detection']
        # nfa_detection_normal(z_denoise, half_win=2, log_nfa_thr=0, sigma=1. / 5, show=1)['detection']
        #
        # nfa_detection_normal(z_denoise, half_win=3, log_nfa_thr=2, sigma=1. / 5, show=2)['detection']
        # nfa_detection_normal(z_denoise, half_win=3, log_nfa_thr=1, sigma=1. / 5, show=1)['detection']
        # nfa_detection_normal(z_denoise, half_win=3, log_nfa_thr=0, sigma=1. / 5, show=1)['detection']


def save_masks(dataset, category):
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    masks_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "masks" / "nfa_thibaud"
    masks_dir.mkdir(parents=True, exist_ok=True)

    kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).type(torch.float32)
    ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32)

    z_paths = natsorted(list(z_dir.glob("*.npy")))
    for z_path in tqdm(z_paths, desc=category):
        z_nlm = torch.from_numpy(np.load(z_path.parent.parent / "z_nlm_p9_d16" / z_path.name))

        nfa = torch.from_numpy(compute_nfa(z_nlm[0].permute(1, 2, 0).cpu().numpy())).permute(2, 0, 1).unsqueeze(0)
        detection = nfa < 3

        detection_post = closing(detection, kernel=ksmall)
        detection_post = torch.from_numpy(median_filter(detection_post, size=3))
        detection_post = dilation(detection_post, kernel=kbig)

        np.save(masks_dir / z_path.name, detection_post.cpu().numpy())


def clustering_detection(z, show=1):
    kernel3 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32)
    kernel5 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32)
    kernel7 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).type(torch.float32)

    z2_sum = (z[0] ** 2).sum(dim=0)
    np_diffs = z2_sum.view(-1, 1).detach().cpu().numpy()

    candidates = {}

    # %% DBSCAN
    from sklearn.cluster import DBSCAN
    clf = DBSCAN()
    clf.fit(np_diffs)
    candidates['dbscan'] = torch.tensor(clf.labels_, device=z.device).view(1, 1, 256, 256) < 0

    detection = 1 * candidates['dbscan']
    detection_post = dilation(detection, kernel=kernel7)
    detection_post = dilation(detection_post, kernel=kernel7)
    candidates['dbscan_post'] = erosion(detection_post, kernel=kernel5)

    if show >= 1:
        plt.imshow(candidates['dbscan'][0, 0].cpu().numpy()), plt.title('DBSCAN'), plt.show()
        plt.imshow(candidates['dbscan_post'][0, 0].cpu().numpy()), plt.title('DBSCAN Post'), plt.show()

    # %% IsolationForest
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest()
    clf.fit(np_diffs)
    candidates['isolation_forest'] = torch.tensor(clf.predict(np_diffs)).view(1, 1, 256, 256) < 0

    detection_post = closing(candidates['isolation_forest'], kernel=kernel5)
    candidates['isolation_forest_post'] = opening(detection_post, kernel=kernel3)

    if show >= 1:
        plt.imshow(candidates['isolation_forest'][0, 0].cpu().numpy()), plt.title('IsolationForest'), plt.show()
        plt.imshow(candidates['isolation_forest_post'][0, 0].cpu().numpy()), plt.title('IsolationForest Post'), plt.show()

    # %% EllipticEnvelope
    from sklearn.covariance import EllipticEnvelope
    clf = EllipticEnvelope()
    clf.fit(np_diffs)
    candidates['elliptic_envelope'] = torch.tensor(clf.predict(np_diffs)).view(1, 1, 256, 256) < 0

    detection_post = closing(candidates['elliptic_envelope'], kernel=kernel5)
    candidates['elliptic_envelope_post'] = opening(detection_post, kernel=kernel3)

    if show >= 1:
        plt.imshow(candidates['elliptic_envelope'][0, 0].cpu().numpy()), plt.title('EllipticEnvelope'), plt.show()
        plt.imshow(candidates['elliptic_envelope_post'][0, 0].cpu().numpy()), plt.title('EllipticEnvelope Post'), plt.show()

    return candidates

    # from sklearn.neighbors import LocalOutlierFactor
    # clf = LocalOutlierFactor(n_neighbors=5)
    # candidates['LocalOutlierFactor'] = torch.tensor(clf.fit_predict(np_diffs)).view(1, 1, 256, 256) < 0
    # if show >= 1:
    #     plt.imshow(candidates['LocalOutlierFactor'][0, 0].cpu().numpy()), plt.title('LocalOutlierFactor'), plt.show()

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


def get_histogram(z):
    z_flat = z.cpu().numpy().ravel()
    bin_edges = [np.percentile(z_flat, 0.1), np.percentile(z_flat, 99.9)]
    bin_edges = [-np.min(np.abs(bin_edges)), np.min(np.abs(bin_edges))]
    pdf, bin_edges = np.histogram(z_flat, 1000, bin_edges, density=False)
    bin_width = np.mean(np.diff(bin_edges))
    bins = (bin_edges + bin_width / 2)[:-1]
    return bins, pdf, bin_width


def gauss_func(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def false_color_img(detection, label):
    assert len(detection.shape) == 3 and detection.shape[0] == 1, "Detection must be of size [1, H, W]"
    assert len(label.shape) == 3 and label.shape[0] == 1, "Label must be of size [1, H, W]"
    return torch.cat([label, 1. * detection, torch.zeros_like(label)], dim=0)


def get_gaussian_fit_candidates(z, z_denoise=None, show=True, n_sigmas=3, theoretical_thr=False, sigma=1):

    bins, pdf, bin_width = get_histogram(z)

    if z_denoise is None:
        bins_denoise, pdf_denoise, bin_width_denoise = bins, pdf, bin_width
    else:
        bins_denoise, pdf_denoise, bin_width_denoise = get_histogram(z_denoise)

    if not theoretical_thr:
        # Executing curve_fit on noisy data
        popt, pcov = curve_fit(gauss_func, bins_denoise, pdf_denoise)
        gain, mean, sigma = popt
        sigma = np.abs(sigma)
        detection = z_denoise.abs() > (mean + n_sigmas * sigma)
    else:
        detection = z_denoise.abs() > (n_sigmas * sigma)

    detection = detection.amax(dim=1, keepdim=True)

    kernel1 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32).to(detection.device)
    kernel2 = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32).to(detection.device)
    detection_post = opening(1 * detection, kernel=kernel1)
    detection_post = dilation(detection_post, kernel=kernel2)
    detection_post = closing(detection_post, kernel=kernel2)

    if show:
        plt.bar(bins, pdf, width=bin_width, alpha=0.5)
        plt.bar(bins_denoise, pdf_denoise, width=bin_width_denoise, alpha=0.5)

        if not theoretical_thr:
            estimation = gauss_func(bins, gain, mean, sigma)
            plt.plot(bins, estimation, c='r')

            n01 = gauss_func(bins, gain, 0, 1)
            plt.plot(bins, n01, c='g')
        else:
            n01 = gauss_func(bins, pdf.max(), 0, 1)
            plt.plot(bins, n01, c='g')

        plt.show()

        plt.imshow(detection[0, 0].cpu().numpy()), plt.title(f'z > {n_sigmas} sigmas'), plt.show()
        plt.imshow(detection_post[0, 0].cpu().numpy()), plt.title(f'z > {n_sigmas} sigmas + morphology'), plt.show()

    return {'detection': detection, 'detection_post': detection_post}


def compute_first_stage_metrics(dataset, category, img_size=256, nlm_patch_size=5):
    z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
    z_paths = natsorted(list(z_dir.glob("*.npy")))

    ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).type(torch.float32)
    kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32)
    from torchmetrics.classification import BinaryAveragePrecision
    auroc = ROC_AUC()
    aupro = AUPRO()
    ap = BinaryAveragePrecision()
    miou = mIoU(thresholds=0.5)
    # iou = IoU(thresholds=0.5)
    iou = IoU(threshold=0.5)
    miou_post = mIoU(thresholds=0.5)
    # iou_post = IoU(thresholds=0.5)
    iou_post = IoU(threshold=0.5)
    for z_path in tqdm(z_paths, desc=category):
        label = load_image_and_label(z_path, img_size)['label']
        label = label.unsqueeze(0) / 255

        # z = torch.from_numpy(np.load(z_path))
        z_nlm = torch.from_numpy(np.load(z_path.parent.parent / "z_nlm_p9_d16" / z_path.name))

        # if SDE == 've':
        #     z = z / 256  # HACK!!

        # z_nlm = torch.from_numpy(denoise_nl_means(
        #     z[0].numpy(),
        #     h=1., sigma=1,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=21,
        #     patch_distance=16, channel_axis=0
        # )).unsqueeze(0)

        anomaly_map = (z_nlm ** 2).sum(dim=1, keepdim=True).sqrt()
        nfa = torch.from_numpy(compute_nfa(z_nlm[0].permute(1, 2, 0).cpu().numpy())).permute(2, 0, 1)
        detection = nfa < 3
        # detection = nfa_detection_normal(z_nlm, log_nfa_thr=log_nfa_thr, half_win=1, sigma=1. / nlm_patch_size, show=0)['detection'][0]
        detection = detection.unsqueeze(0) if len(detection.shape) == 3 else detection

        detection_post = torch.from_numpy(median_filter(detection, size=3))
        # detection_post = closing(detection, kernel=kbig)
        detection_post = opening(1 * detection_post, kernel=ksmall)
        # pad detection_post with 1 pixel
        detection_post = F.pad(detection_post, (1, 1, 1, 1), mode='constant', value=0)
        detection_post = dilation(detection_post, kernel=kbig)
        detection_post = detection_post[:, :, 1:-1, 1:-1]

        auroc.update((anomaly_map.ravel(), label.ravel() > 0.5))
        aupro.update(anomaly_map, label > 0.5)
        ap.update(anomaly_map, label > 0.5)

        miou.update(detection.float(), label.float())
        # miou.update(torch.cat([detection.float(), label.float()]).amax(dim=0, keepdim=True), label.float())
        iou.update(detection.float(), label.float())

        miou_post.update(detection_post.float().clone(), label.float())
        iou_post.update(detection_post.float(), label.float())

    print(
        f"{category} | "
        f"AUROC: {auroc.compute().item() * 100:.2f}, "
        f"AUPRO: {aupro.compute().item() * 100:.2f}, "
        f"AP: {ap.compute().item() * 100:.2f}, "
        f"mIoU: {miou.compute().item() * 100:.2f}, "
        f"mIoU post: {miou_post.compute().item() * 100:.2f}",
        f"IoU: {iou.compute().item() * 100:.2f}, "
        f"IoU post: {iou_post.compute().item() * 100:.2f}"
    )


def generate_ode_chart():

    import models.utils as mutils

    config = configs.get_config()
    ad = DAD(config)
    params = mutils.get_ddpm_params(config)

    x0 = torch.cat([
        torch.randn(100000) * 0.5 + 0.2,
        torch.randn(100000) * 0.6 + 2,
        torch.randn(100000) * 0.4 - 1,
    ])
    ts = torch.from_numpy(np.linspace(0, 1, 1000))
    x = [x0]
    for t in ts:
        x.append(params['sqrt_alphas_cumprod'][int(t * 1000) - 1] * x0 + params['sqrt_1m_alphas_cumprod'][int(t * 1000) - 1] * torch.randn_like(x0))
    x = torch.stack(x).T

    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 1000)
    # bins = np.linspace(-5, 5, 1000)
    density = []
    for i in range(x.shape[1]):
        pdf, _ = np.histogram(x[:, i].numpy(), bins, density=False)
        density.append(pdf)
    density = np.stack(density).T

    plt.imshow(density)
    plt.show()


def main():

    # generate_ode_chart()

    dataset = "mvtec"

    # print_likelihood_masks_metrics(dataset)
    # return

    for category in ALL_CATEGORIES[dataset]:  # ["screw"]:  # ["carpet", "grid", "bottle"]:  # ALL_CATEGORIES[dataset]:

        # save_z(dataset, category)
        # save_z_nlm(dataset, category)

        # uflow_comparison(dataset, category)

        # compute_first_stage_metrics(dataset, category)

        # test_likelihood_masks(dataset, category)

        # save_masks(dataset, category)

        # inpainting(dataset, category)

        compute_anomaly_maps(dataset, category)

        #
        # Up to here july 2024
        #

        # save_likelihood_masks_grid(dataset, category)
        # save_likelihood_masks_grid_nfa(dataset, category)
        # save_likelihood_masks_grid_new(dataset, category)

        # save_final_likelihood_masks_nfa(dataset, category)
        # save_final_likelihood_masks(dataset, category)

        # refine_postprocessing(dataset, category)
        # refine_postprocessing_nfa(dataset, category


if __name__ == '__main__':
    main()
