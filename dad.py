from skimage.restoration import denoise_nl_means
import lightning as L
import numpy as np
import torch
from torchvision.utils import make_grid
from ignite.contrib.metrics import ROC_AUC

import datasets
import losses
import sampling
import sde_lib
from likelihood import get_likelihood_fn
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn
from models import ncsnpp, utils as mutils
from models.ema import ExponentialMovingAverage
from sde_lib import VESDE, VPSDE
from utils import l2
from metrics.mIoU import mIoU
from nfa import nfa_detection_normal


class DAD(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.validation_step_outputs = []
        self.num_debug_epochs = 200

        self.scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)

        self.score_model, self.state = self.setup_model()
        self.sde, self.sampling_eps = self.setup_sde()
        self.loss_fn = self.setup_loss_fn()
        self.sampling_fn, self.sampling_shape = self.setup_sampling_fn()
        self.likelihood_fn = get_likelihood_fn(self.sde, self.inverse_scaler, eps=1e-5)

        # self.load_state_dict(torch.load("/home/data/tai/phd/training/tmp/ve/mvtec/carpet/exp_0002/ckpts/VAL_epoch=355_loss-val_loss=3023.31494.ckpt")["state_dict"])

    def setup_model(self):
        score_model = mutils.create_model(self.config)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=self.config.model.ema_rate)
        optimizer = losses.get_optimizer(self.config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        return score_model, state

    def setup_sde(self):
        if self.config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(
                beta_min=self.config.model.beta_min, beta_max=self.config.model.beta_max, N=self.config.model.num_scales
            )
            sampling_eps = 1e-3
        elif self.config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(
                beta_min=self.config.model.beta_min, beta_max=self.config.model.beta_max, N=self.config.model.num_scales
            )
            sampling_eps = 1e-3
        elif self.config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(
                sigma_min=self.config.model.sigma_min, sigma_max=self.config.model.sigma_max,
                N=self.config.model.num_scales
            )
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.config.training.sde} unknown.")
        return sde, sampling_eps

    def setup_loss_fn(self):
        continuous = self.config.training.continuous
        reduce_mean = self.config.training.reduce_mean
        likelihood_weighting = self.config.training.likelihood_weighting
        train = True

        if continuous:
            loss_fn = get_sde_loss_fn(
                self.sde, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting
            )
        else:
            assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, VESDE):
                loss_fn = get_smld_loss_fn(self.sde, train, reduce_mean=reduce_mean)
            elif isinstance(self.sde, VPSDE):
                loss_fn = get_ddpm_loss_fn(self.sde, train, reduce_mean=reduce_mean)
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")

        return loss_fn

    def setup_sampling_fn(self):
        if self.config.training.snapshot_sampling:
            sampling_shape = (
                self.config.training.batch_size, self.config.data.num_channels,
                self.config.data.image_size, self.config.data.image_size
            )
            sampling_fn = sampling.get_sampling_fn(
                self.config, self.sde, sampling_shape, self.inverse_scaler,
                self.sampling_eps
            )
            return sampling_fn, sampling_shape
        else:
            return None, None

    def step(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.from_numpy(batch._numpy()).float()
            batch = batch.permute(0, 3, 1, 2)
        batch = self.scaler(batch)
        loss = self.loss_fn(self.score_model, batch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch['image'])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].shape[0], sync_dist=True)
        self.logger.experiment.add_scalar("00_Loss/Train", loss, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch['image'])
        self.log("val_loss", loss, batch_size=batch['image'].shape[0], sync_dist=True)
        self.logger.experiment.add_scalar("00_Loss/Val", loss, self.current_epoch)

        if (self.current_epoch % self.num_debug_epochs == 0) and (self.current_epoch != 0):
            if batch_idx == 0:  # Only because it takes to long to compute over the whole db
                bpd, z, nfe = self.likelihood_fn(self.score_model, batch['image'])
                l2_norm = l2(z).repeat(1, 3, 1, 1)
                self.validation_step_outputs.append(
                    {'image': batch['image'], 'label': batch['target'].repeat(1, 3, 1, 1), 'z': l2_norm}
                )
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch == 0:
            self.log("val_auroc", 0, sync_dist=True)
            self.log("val_miou", 0, sync_dist=True)
            return

        if (self.current_epoch % self.num_debug_epochs == 0) and (self.current_epoch != 0):

            # Generate some samples
            sample, n = self.sampling_fn(self.score_model)
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid(sample, nrow, padding=2)
            self.logger.experiment.add_image(
                f"Generated Samples",
                image_grid,
                self.current_epoch,
                dataformats="CHW"
            )

            # Generate some likelihoods
            # TODO: Add detections here
            all_images = []
            zs_nlm = []
            labels = []
            for batch in self.validation_step_outputs:
                for i in range(batch['image'].shape[0]):
                    z = batch['z'][i]
                    z_nlm = torch.from_numpy(denoise_nl_means(
                        z.cpu().numpy() / (256 if self.config.training.sde.lower() == 'vesde' else 1),
                        h=1., sigma=1,
                        fast_mode=False, preserve_range=True,
                        patch_size=5, patch_distance=6, channel_axis=0
                    )).to(batch['z'].device)
                    zs_nlm.append(z_nlm)
                    labels.append(batch['label'][i])
                    z = z - z.min()
                    z /= z.max()
                    z_nlm = z_nlm - z_nlm.min()
                    z_nlm /= z_nlm.max()
                    all_images.extend([batch['image'][i], z, z_nlm, batch['label'][i]])
            image_grid = make_grid(all_images, 4, padding=2)
            self.logger.experiment.add_image(
                f"Likelihoods (Zs)",
                image_grid,
                self.current_epoch,
                dataformats="CHW"
            )

            # Compute and log AUROC
            auroc = ROC_AUC()
            miou = mIoU(thresholds=0.5)
            for z_nlm, label in zip(zs_nlm, labels):
                anomaly_map = l2(z_nlm)
                nfa_detection = nfa_detection_normal(z_nlm.unsqueeze(0), 0, 0, sigma=1. / 5, show=0)['detection']
                auroc.update((anomaly_map.ravel(), label[0].ravel() > 0.5))
                miou.update(nfa_detection.float().to(self.config.device), label[:1].unsqueeze(0).float())
            auroc_val = float(auroc.compute())
            miou_val = float(miou.compute())
            self.log_dict(
                {"val_auroc": auroc_val, "val_miou": miou_val},
                on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=len(zs_nlm), sync_dist=True
            )
            self.logger.experiment.add_scalar("01_METRICS/AUROC_Val", auroc_val, self.current_epoch)
            self.logger.experiment.add_scalar("01_METRICS/mIoU_Val", miou_val, self.current_epoch)

            self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, 0.999),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )
        return optimizer
