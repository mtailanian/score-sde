import numpy as np
import torch
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
import lpips
from torch import nn
from tqdm import tqdm


class MetaLpips(nn.Module):
    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.l_pips_sq = (
            lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True)
        )

    def lpips_loss(self, ph_img, anomaly_img, mode=0):
        def ano_cam(ph_img_, anomaly_img_, mode=0):
            anomaly_img_.requires_grad_(True)
            loss_lpips = self.l_pips_sq(anomaly_img_, ph_img_, normalize=True, retPerLayer=False)
            return loss_lpips.cpu().detach().numpy()

        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        ano_map = ano_cam(ph_img, anomaly_img, mode)
        return ano_map

    def get_saliency(self, x_rec, x):
        saliency = self.lpips_loss(x_rec, x)
        saliency = gaussian_filter(saliency, sigma=2)
        return saliency

    def forward(self, x_rec, x):

        assert len(x_rec.shape) == 4 and len(x.shape) == 4, "Input images should be of shape (B, C, H, W)"

        l1s, lpips, meta_lpips = [], [], []
        if self.verbose:
            progress_bar = tqdm(zip(x_rec, x), total=x_rec.shape[0], desc='Computing anomaly maps', leave=False)
        else:
            progress_bar = zip(x_rec, x)
        for rec, orig in progress_bar:
            anomaly_map = self._forward(rec.unsqueeze(0), orig.unsqueeze(0))
            l1s.append(anomaly_map['L1'])
            lpips.append(anomaly_map['LPIPS'])
            meta_lpips.append(anomaly_map['META-LPIPS'])

        l1 = np.concatenate(l1s).sum(axis=1).mean(axis=0)
        lpips = np.concatenate(lpips).mean(axis=0)[0]
        meta_lpips = np.concatenate(meta_lpips).mean(axis=(0, 1))

        return {'L1': l1, 'LPIPS': lpips, 'META-LPIPS': meta_lpips}

    def _forward(self, x_rec, x):
        """
        Compute residual between two images.
        Args:
            x_rec: Reconstructed image, with shape (1, C, H, W)
            x: Original image, with shape (1, C, H, W)
        """

        self.l_pips_sq = self.l_pips_sq.to(x.device)
        # Lpips
        x_rec = torch.clamp(x_rec, 0, 1)
        x = torch.clamp(x, 0, 1)
        saliency = self.get_saliency(x_rec, x)  # lpips

        # Lpips over the equalized images
        x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
        x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
        saliency2 = self.get_saliency(
            torch.Tensor(x_rec_rescale).to(x_rec.device),
            torch.Tensor(x_rescale).to(x_rec.device)
        )

        saliency = saliency * saliency2  # Lpips final

        # L1 difference between the equalized images
        x_res = np.abs(x_rec_rescale - x_rescale)

        # The final anomaly maps would be the product x_res * saliency
        return {'L1': x_res, 'LPIPS': saliency, 'META-LPIPS': x_res * saliency}
