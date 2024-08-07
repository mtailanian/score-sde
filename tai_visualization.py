import torch
import numpy as np
import torch
import cv2
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib as mpl
from skimage.restoration import estimate_sigma, denoise_nl_means
from scipy.ndimage import median_filter
from torchvision.utils import draw_segmentation_masks
from kornia.morphology import dilation, closing
import plotly.graph_objects as go

import datasets
from datasets_anomaly import ALL_CATEGORIES
from utils import imshow_tensor, l2, load_image_and_label
from path_utils import get_path
from residual_whitening import compute_nfa
from anomaly_detector import AnomalyDetector
from meta_lpips import MetaLpips


CKPT = 10
SDE = 've'
DATASET = 'mvtec'

if SDE == 've':
    from configs.ve import anomaly_256_ncsnpp_continuous as configs
elif SDE == 'subvp':
    from configs.subvp import anomaly_256_ncsnpp_continuous as configs


def generate_labels_image(labels, width, height=50, padding=2, rotate=True, font_scale=1.3):
    font = cv2.FONT_HERSHEY_SIMPLEX

    images = [np.ones((height, padding, 3), dtype=np.uint8) * 255]
    if rotate:
        labels = labels[::-1]
    for l in labels:
        textsize = cv2.getTextSize(l, font, font_scale, 2)[0]
        textX = int(width // 2 - (textsize[0] / 2))
        textY = int(height // 2 + (textsize[1] / 2))
        images.append(cv2.putText(np.ones((height, width, 3), dtype=np.uint8) * 255, l, (textX, textY), font, font_scale, (0, 0, 0), 2))
        images.append(np.ones((height, padding, 3), dtype=np.uint8) * 255)
    label_img = np.concatenate(images, axis=1)
    if rotate:
        label_img = cv2.rotate(label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return torch.tensor(label_img).permute(2, 0, 1)


def generate_samples_grid():
    config = configs.get_config()
    config.data.dataset = DATASET
    all_images = []
    for category in ALL_CATEGORIES[DATASET]:
        config.data.category = category
        config.training.batch_size = 1

        train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
        train_img = next(iter(train_ds))['image']

        gen = np.load(f"/home/data/tai/phd/training/dad/ve/{DATASET}/{category}/samples/iter_100001/sample.np")
        gen = torch.tensor(gen).permute(0, 3, 1, 2)
        all_images.append(torch.cat([train_img, gen[:8] / 255.]))

    grid = make_grid(torch.cat(all_images, dim=0), nrow=9, padding=4, pad_value=1)
    labels_img = generate_labels_image(
        ['Real image', 'Gen #1', 'Gen #2', 'Gen #3', 'Gen #4', 'Gen #5', 'Gen #6', 'Gen #7', 'Gen #8'],
        256, 50, 4,
        rotate=False
    ) / 255.
    grid = torch.concat([labels_img, grid], dim=1)

    imshow_tensor(grid)
    out_path = get_path('results') / 'dad' / SDE / "samples"
    save_image(grid, out_path / f"{DATASET}.pdf")


def apply_colormap(tensor, cmap='turbo'):
    colormap = mpl.colormaps[cmap]
    return torch.from_numpy(
        colormap(tensor[0].mul(255).add_(0.5).clamp_(0, 255).to('cpu').numpy().astype(np.uint8))[:, :, :3]).permute(2, 0, 1)


def post_process_detection(detection):
    kbig = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).type(torch.float32)
    ksmall = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).type(torch.float32)

    detection_post = closing(detection, kernel=ksmall)
    detection_post = torch.from_numpy(median_filter(detection_post, size=3))
    detection_post = dilation(detection_post, kernel=kbig)
    return detection_post


def generate_likelihoods_grid():
    def get_normalized_ll(zi):
        lli = l2(zi)
        lli = lli - torch.quantile(lli, 0.01)
        lli = lli / torch.quantile(lli, 0.99)
        lli = 1 - lli
        lli = torch.clip_(lli, 0, 1)
        return lli

    dataset = "mvtec"
    chosen_images = {
        "carpet": "color_000",
        "grid": "bent_003",
        "bottle": "broken_large_004",
        "hazelnut": "crack_003",
    }

    all_images = []
    for category, img_id in chosen_images.items():

        # Original embedding Z
        z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
        z_path = list(z_dir.glob(f"*{img_id}.npy"))[0]
        z = torch.from_numpy(np.load(z_path))[0]
        c = z.shape[0]

        ll = get_normalized_ll(z)
        ll = apply_colormap(ll[0], 'jet')

        # Load original image and label
        images = load_image_and_label(z_path, z.shape[-1], fill=False)
        image = images['image_with_label'] / 255.
        label = images['label'].repeat(c, 1, 1) > 127

        # Denoised embedding Z
        # sigma_est = np.mean(estimate_sigma(z.cpu().numpy(), channel_axis=0))
        # sigma_est = configs.get_config().model.sigma_max
        # z_nlm = torch.from_numpy(denoise_nl_means(
        #     z.cpu().numpy(),
        #     # h=1.15 * sigma_est, sigma=sigma_est,
        #     h=0.8 * sigma_est, sigma=sigma_est,
        #     fast_mode=False, preserve_range=True,
        #     patch_size=9, patch_distance=16, channel_axis=0,
        #     # patch_size=5, patch_distance=11, channel_axis=0
        # )).unsqueeze(0)
        z_nlm = torch.from_numpy(np.load(z_path.parent.parent / "z_nlm_p9_d16" / z_path.name))
        ll_nlm = get_normalized_ll(z_nlm)
        ll_nlm = apply_colormap(ll_nlm[0], 'jet')

        # Detection
        nfa = torch.from_numpy(compute_nfa(z_nlm[0].permute(1, 2, 0).cpu().numpy())).permute(2, 0, 1)
        detection = nfa < 3
        detection_post = post_process_detection(detection.unsqueeze(0))[0]
        ## Draw borders
        border = (dilation(label.unsqueeze(0).float(), kernel=torch.ones(5, 5)) - label.unsqueeze(0).float()) == 1
        detection = detection.type(torch.uint8).repeat(c, 1, 1) * 255
        detection = draw_segmentation_masks(detection, border[0], colors='yellow', alpha=0.7) / 255.
        detection_post = detection_post.type(torch.uint8).repeat(c, 1, 1) * 255
        detection_post = draw_segmentation_masks(detection_post, border[0], colors='yellow', alpha=0.7) / 255.

        # Inpainting
        inpainted_path = inpainting_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "inpainting" / "nfa_thibaud" / z_path.name
        inpainted = torch.from_numpy(np.load(inpainted_path))
        # ckpt_path = str(get_path('training') / 'dad' / SDE / dataset / category / 'checkpoints' / f'checkpoint_{CKPT}.pth')
        # ad = AnomalyDetector(ckpt_path)
        # inpainted = ad.inpainter(ad.score_model, ad.scaler(images['image'].unsqueeze(0).cuda() / 255.), 1 - detection_post[:1].unsqueeze(0).cuda())[0].cpu()

        # Difference
        lpips = MetaLpips(verbose=False)
        anomaly_map = torch.from_numpy(lpips(inpainted.unsqueeze(0), image.unsqueeze(0))['LPIPS']).unsqueeze(0)
        anomaly_map_cm = anomaly_map - torch.quantile(anomaly_map, 0.001)
        anomaly_map_cm = anomaly_map_cm / torch.quantile(anomaly_map_cm, 0.999)
        anomaly_map_cm = apply_colormap(anomaly_map_cm, 'jet')

        all_images.append(torch.stack([image, ll, ll_nlm, detection, detection_post, inpainted, anomaly_map_cm, 1. * label]))
        # all_images.append(torch.stack([image, inpainted, anomaly_map_cm, 1. * label]))

    all_images = torch.stack(all_images, dim=1)
    all_images = torch.flatten(all_images, 0, 1)
    grid = make_grid(all_images, nrow=len(chosen_images), padding=4, pad_value=1)

    labels_img = generate_labels_image(
        ['Original', 'Raw log-likelihood', 'NLM log-likelihood', 'Detection', 'Det. Post-Proc.', 'Inpainted', 'Anomaly Map', 'Label'],
        # ['Original', 'Inpainted', 'Anomaly Map', 'Label'],
        256, 30, 4,
        rotate=True,
        font_scale=0.8
    ) / 255.
    grid = torch.concat([labels_img, grid], dim=2)

    imshow_tensor(grid)
    save_image(grid, "results/likelihoods_13.pdf")


def generate_histograms():

    dataset = "mvtec"
    chosen_images = {
        "carpet": "color_000",
        "grid": "bent_003",
        "bottle": "broken_large_004",
        "hazelnut": "crack_003",
    }

    all_images = []
    for category, img_id in chosen_images.items():

        # Embedding Z
        z_dir = get_path('results') / "dad" / SDE / dataset / category / f"ckpt_{CKPT}" / "z"
        z_path = list(z_dir.glob(f"*{img_id}.npy"))[0]

        z = np.load(z_path)
        z_nlm = np.load(z_path.parent.parent / "z_nlm_p9_d16" / z_path.name)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=z.flatten(), name='Raw log-likelihood', histnorm='probability density'))
        fig.add_trace(go.Histogram(x=z_nlm.flatten(), name='NLM log-likelihood', histnorm='probability density'))

        # Overlay both histograms
        fig.update_layout(
            width=1024,
            height=512,
            barmode='overlay',
            xaxis_title_text='Value',
            yaxis_title_text='Probability density',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        )
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        fig.update_xaxes(range=[-1500, 1500])
        fig.update_yaxes(type='log')
        fig.write_image(f"results/hist_{category}_{z_path.stem}.pdf")


def main():
    # generate_samples_grid()
    generate_likelihoods_grid()
    # generate_histograms()


if __name__ == '__main__':
    main()
