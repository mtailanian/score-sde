import os.path
from typing import Union
from torch import nn
import torch
from natsort import natsorted
from PIL import Image
from pathlib import Path
from torchvision import transforms
from einops import repeat
import matplotlib.pyplot as plt
import kornia as K

from controllable_generation import get_pc_inpainter
from sampling import ReverseDiffusionPredictor, LangevinCorrector, EulerMaruyamaPredictor
from sde_lib import VESDE, subVPSDE
from models.ncsnpp import NCSNpp
import datasets
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, show_samples
from losses import get_optimizer
from likelihood import get_likelihood_fn
from meta_lpips import MetaLpips
from chosen_checkpoints import ckpts
from path_utils import get_path
from sampling import get_pc_sampler
from utils import imshow_tensor


class AnomalyDetector(nn.Module):
    def __init__(self, ckpt_path, config=None):
        super(AnomalyDetector, self).__init__()

        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        if config is None:
            if "/ve/" in ckpt_path:
                from configs.ve import anomaly_256_ncsnpp_continuous as configs
            elif "/subvp/" in ckpt_path:
                from configs.subvp import anomaly_256_ncsnpp_continuous as configs
            else:
                raise ValueError(f"Unknown SDE in checkpoint path: {ckpt_path}")
            self.config = configs.get_config()
        else:
            self.config = config

        self.score_model, self.inpainter, self.likelihood_fn, self.sampling_fn, self.scaler = self.setup_model(ckpt_path)

        self.transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor(),
        ])

        self.anomaly_map_generator = MetaLpips(verbose=False)

    def setup_model(self, ckpt_path):
        if self.config.training.sde == 'vesde':
            sde = VESDE(
                sigma_min=self.config.model.sigma_min,
                sigma_max=self.config.model.sigma_max,
                N=self.config.model.num_scales
            )
            sampling_eps = 1e-5
        elif self.config.training.sde == 'subvpsde':
            sde = subVPSDE(
                beta_min=self.config.model.beta_min,
                beta_max=self.config.model.beta_max,
                N=self.config.model.num_scales
            )
            sampling_eps = 1e-3
        else:
            raise ValueError(f"Unknown SDE: {self.config.training.sde}")

        score_model = mutils.create_model(self.config)
        optimizer = get_optimizer(self.config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=self.config.model.ema_rate)
        state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
        state = restore_checkpoint(ckpt_path, state, self.config.device)
        ema.copy_to(score_model.parameters())

        if self.config.sampling.predictor == "euler_maruyama":
            predictor = EulerMaruyamaPredictor
        elif self.config.sampling.predictor == "reverse_diffusion":
            predictor = ReverseDiffusionPredictor

        if self.config.sampling.corrector == "none":
            corrector = None
        elif self.config.sampling.corrector == "langevin":
            corrector = LangevinCorrector

        snr = 0.16  # @param {"type": "number"}
        n_steps = 1  # @param {"type": "integer"}
        probability_flow = False  # @param {"type": "boolean"}
        scaler = datasets.get_data_scaler(self.config)
        inverse_scaler = datasets.get_data_inverse_scaler(self.config)

        inpainter = get_pc_inpainter(
            sde,
            predictor, corrector,
            inverse_scaler,
            snr=snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=self.config.training.continuous,
            denoise=True
        )

        likelihood_fn = get_likelihood_fn(sde, inverse_scaler, eps=1e-5)

        shape = (1, self.config.data.num_channels, self.config.data.image_size, self.config.data.image_size)
        sampling_fn = get_pc_sampler(
            sde, shape, predictor, corrector, inverse_scaler, snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=self.config.training.continuous,
            eps=sampling_eps,
            device=self.config.device
        )

        return score_model, inpainter, likelihood_fn, sampling_fn, scaler

    def forward(self, img_path: Union[str, Path], n_samples=1):
        img = self.transform(Image.open(str(img_path)).convert('RGB')).unsqueeze(0).to(self.config.device)
        img = repeat(img, "1 ... -> n ...", n=n_samples)
        plt.imshow(img[0].permute(1, 2, 0).cpu().numpy())
        plt.show()

        # Compute likelihood
        print('Computing likelihood...', end=" ")
        bpd, z, nfe = self.likelihood_fn(self.score_model, img)
        print("Done!")

        # Create low likelihood mask
        zi = z.abs().sum(dim=1, keepdims=True)
        zi = K.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(15, 15))(zi)
        plt.imshow(zi[0, 0].cpu().numpy())
        plt.show()

        thr = (zi.min() + zi.max()) / 2
        mask = zi > thr
        mask = K.morphology.dilation(
            1 * mask,
            kernel=torch.ones(15, 15).to(mask.device),
            border_type='reflect'
        )
        plt.imshow(mask[0, 0].cpu().numpy())
        plt.show()

        # Inpaint low likelihood mask
        print('Inpainting...', end=" ")
        inpainted = self.inpainter(self.score_model, self.scaler(img), 1 - mask)
        print("Done!")
        plt.imshow(inpainted[0].permute(1, 2, 0).cpu().numpy())
        plt.show()

        # Compute anomaly map
        anomaly_maps = self.anomaly_map_generator(inpainted, img)
        plt.imshow(anomaly_maps['LPIPS'])
        plt.show()

        return anomaly_maps['LPIPS']


def main(sde="subvp", ckpt=10, dataset="visa", category="candle"):
    ckpt_path = str(get_path('training') / 'dad' / sde / dataset / category / 'checkpoints' / f'checkpoint_{ckpt}.pth')
    image_paths = natsorted(list((get_path('data') / dataset).glob(f"{category}/test/*/*")))

    ad = AnomalyDetector(ckpt_path)

    # anomaly_map = ad(image_paths[0])
    x, n = ad.sampling_fn(ad.score_model)
    imshow_tensor(x)


if __name__ == '__main__':
    main()
