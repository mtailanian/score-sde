import torch
import tensorflow as tf
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
from path_utils import get_path
from kornia.morphology import dilation


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def image_grid(x, config):
  size = config.data.image_size
  channels = config.data.num_channels
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img


def show_samples(x, config):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x, config)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.show()


def load_image_and_label(z_path: Path, size=256, fill=True, alpha=0.5, category_position_in_path=-4):
    data_dir = get_path('data')
    resize_transform = torchvision.transforms.Resize(size, antialias=True)
    resize_label_transform = torchvision.transforms.Resize(size, antialias=True, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    category = z_path.parts[category_position_in_path]
    # category = z_path.parent.parent.parent.stem
    defect_type = "_".join(z_path.stem.split("_")[:-1])
    name = Path(z_path.stem).stem.split("_")[-1]

    image = resize_transform(torchvision.io.read_image(str(data_dir / "mvtec" / category / "test" / defect_type / f"{name}.png")))
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if defect_type == 'good':
        label = torch.zeros([1, image.shape[-2], image.shape[-1]])
    else:
        label = resize_label_transform(torchvision.io.read_image(str(data_dir / "mvtec" / category / "ground_truth" / defect_type / f"{name}_mask.png")))

    annotated = image.clone()
    border = label > 127
    if not fill:
        border = border.unsqueeze(0)
        border = (dilation(border.float(), kernel=torch.ones(5, 5)) - border.float()) == 1
        border = border[0]
    annotated = torchvision.utils.draw_segmentation_masks(annotated, border, colors='yellow', alpha=alpha)

    return {'image': image, 'label': label, 'image_with_label': annotated}


def generate_labels_image(labels, width, height=50, padding=2, font_scale=1.3):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    images = [np.ones((height, padding, 3), dtype=np.uint8) * 255]
    for l in labels[::-1]:
        textsize = cv2.getTextSize(l, font, font_scale, 2)[0]
        textX = int(width // 2 - (textsize[0] / 2))
        textY = int(height // 2 + (textsize[1] / 2))
        images.append(cv2.putText(np.ones((height, width, 3), dtype=np.uint8) * 255, l, (textX, textY), font, font_scale, (0, 0, 0), 2))
        images.append(np.ones((height, padding, 3), dtype=np.uint8) * 255)
    label_img = np.concatenate(images, axis=1)
    label_img = cv2.rotate(label_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return torch.tensor(label_img).permute(2, 0, 1)


def add_blank_row(img, positions, blank_size, img_size, padding):
    for row in reversed(positions):
        y = row * (img_size + padding)
        img = torch.cat([img[:, :y, :], torch.ones(img.shape[0], blank_size, img.shape[2]), img[:, y:, :]], dim=1)
    return img


def imshow_tensor(image, title=None):
    if len(image.shape) == 4:
        image = image[0]
    if image.shape[0] == 1:
        plt.imshow(image[0].detach().cpu().numpy())
    else:
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())

    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def l2(image):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return torch.sqrt(torch.sum(image ** 2, dim=1, keepdim=True))
