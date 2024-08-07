import os
from glob import glob
from pathlib import Path

import yaml
import numpy as np
import torch.utils.data
from PIL import Image
import lightning as L

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from path_utils import get_path

ALL_CATEGORIES = {
    'mvtec': [
        "carpet", "grid", "leather", "tile", "wood",
        "bottle", "cable", "capsule", "hazelnut", "metal_nut",
        "pill", "screw", "toothbrush", "transistor", "zipper"
    ],
    'visa': [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2",
        "pcb1", "pcb2", "pcb3", "pcb4",
        "pipe_fryum",
    ],
}


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class AnomalyDatamodule(L.LightningDataModule):
    def __init__(self, config):
        self.dataset = config.data.dataset
        self.category = config.data.category
        self.input_size = config.data.image_size
        self.batch_size = config.training.batch_size

        self.train_dataset = AnomalyDataset(config.data.dataset, self.category, self.input_size, is_train=True)
        self.val_dataset = AnomalyDataset(config.data.dataset, self.category, self.input_size, is_train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=False,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False,
            worker_init_fn=worker_init_fn
        )


class AnomalyDataset(Dataset):
    def __init__(self, dataset, category, input_size, is_train):
        self.dataset = dataset
        if self.dataset == "mvtec":
            self.data_dir = str(get_path("data") / "mvtec")
            self.img_ext, self.mask_ext = "png", "png"
        elif self.dataset == "visa":
            self.data_dir = str(get_path("data") / "visa" / "visa_pytorch")
            self.img_ext, self.mask_ext = "JPG", "png"
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not yet supported.")

        self.is_train = is_train
        transform = [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
        # if is_train:
        # 	transform.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(transform)

        if is_train:
            self.image_files = list(glob(os.path.join(self.data_dir, category, "train", "good", f"*.{self.img_ext}")))
        else:
            self.image_files = sorted(glob(os.path.join(self.data_dir, category, "test", "*", f"*.{self.img_ext}")))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)

        if self.is_train:
            return {'image': image}
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target_path = image_file.replace("test", "ground_truth")
                if self.dataset == 'mvtec':
                    target_path = target_path.replace(f".{self.img_ext}", f"_mask.{self.mask_ext}")
                elif self.dataset == 'visa':
                    target_path = target_path.replace(f".{self.img_ext}", f".{self.mask_ext}")
                target = Image.open(target_path).convert('L')
                target = self.transform(target)
            return {'image': image, 'target': target, 'image_path': image_file}

    def __len__(self):
        return len(self.image_files)
