import torch
from natsort import natsorted
from PIL import Image
from pathlib import Path
from torchvision import transforms
from einops import repeat
import matplotlib.pyplot as plt
import kornia as K

from controllable_generation import get_pc_inpainter
from sampling import ReverseDiffusionPredictor, LangevinCorrector
from sde_lib import VESDE
from configs.ve import anomaly_256_ncsnpp_continuous as configs
from models.ncsnpp import NCSNpp
import datasets
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, show_samples
from losses import get_optimizer
from likelihood import get_likelihood_fn


category = "grid"
ckpt_filename = f"/data/tai/phd/training/sde/{category}/checkpoints/checkpoint_61.pth"
config = configs.get_config()  

sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
score_model = mutils.create_model(config)
optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)  
state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.16 #@param {"type": "number"}
n_steps = 1 #@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

pc_inpainter = get_pc_inpainter(
    sde,
    predictor, corrector,
    inverse_scaler,
    snr=snr,
    n_steps=n_steps,
    probability_flow=probability_flow,
    continuous=config.training.continuous,
    denoise=True
)

image_paths = natsorted(list(Path("/data/tai/phd/data/mvtec").glob(f"{category}/test/*/*.png")))
transform = transforms.Compose([
    transforms.Resize(config.data.image_size),
    transforms.ToTensor(),
])

img = transform(Image.open(image_paths[0]).convert('RGB')).unsqueeze(0).to(config.device)
img = repeat(img, "1 ... -> n ...", n=1)
show_samples(img, config)

mask = torch.ones_like(img)
mask[:, :, 60:100, 50:80] = 0.
mask[:, :, 90:120, 180:220] = 0.
show_samples(img * mask, config)

x = pc_inpainter(score_model, scaler(img), mask)
show_samples(x, config)

# %%

likelihood_fn = get_likelihood_fn(sde, inverse_scaler, eps=1e-5)
bpd, z, nfe = likelihood_fn(score_model, img[:1])

plt.imshow(z.mean(dim=1)[0].cpu().numpy())
plt.show()

plt.imshow(z.abs().sum(dim=1)[0].cpu().numpy())
plt.show()

plt.imshow((z ** 2).sum(dim=1).sqrt()[0].cpu().numpy())
plt.show()

log_prob_i = -(z ** 2).mean(dim=1) * 0.5
prob_i = torch.exp(log_prob_i)
plt.imshow(prob_i[0].cpu().numpy())
plt.show()

# %%

zi = z.abs().sum(dim=1, keepdims=True)
zif = K.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(5, 5))(zi)

plt.imshow(zi[0, 0].cpu().numpy() > float((zi.min() + zi.max()) / 2))
plt.show()

thr = (zif.min() + zif.max()) / 2
mask = zif > thr
plt.imshow(mask[0, 0].cpu().numpy())
plt.show()

kernel = torch.ones(5, 5).to(mask.device)
mask2 = K.morphology.dilation(1 * mask, kernel=kernel, border_type='reflect')
plt.imshow(mask2[0, 0].cpu().numpy())
plt.show()

