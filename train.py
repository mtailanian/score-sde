"""
For monitoring with tensorboard, first ssh to the server redirecting ports:
`ssh -L 16006:127.0.0.1:6006 dsense@michigan`

Then, run tensorboard in the server:
`conda activate scoliosis`
`tensorboard --logdir /path/to/logs`

Finally, in the local machine, open the browser and go to:
`http://127.0.0.1:16006`
"""

import argparse
import random
import warnings

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import datasets
from path_utils import prepare_training_dir
from dad import DAD

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def get_config(sde):
    if sde == "ve":
        import configs.ve.anomaly_256_ncsnpp_continuous as configs
    elif sde == "vp":
        import configs.vp.anomaly_256_ncsnpp_continuous as configs
    elif sde == "subvp":
        import configs.subvp.anomaly_256_ncsnpp_continuous as configs
    else:
        raise ValueError(f"Unknown SDE: {sde}")
    return configs.get_config()


class DataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scaler = datasets.get_data_scaler(self.config)
        self.inverse_scaler = datasets.get_data_inverse_scaler(self.config)
        self.train_dataload, self.val_dataload, _ = datasets.get_dataset(self.config, self.config.data.uniform_dequantization)

    def train_dataloader(self):
        return self.train_dataload
        # return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size, num_workers=8)

    def val_dataloader(self):
        return self.val_dataload
        # return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, num_workers=8)


def main(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    config = get_config(args.sde)
    config.data.dataset = args.dataset
    config.data.category = args.category

    logs_dir, ckpts_dir = prepare_training_dir(subfolder=f"tmp/{args.sde}/{args.dataset}/{args.category}")
    logger = TensorBoardLogger(logs_dir)

    # Model
    model = DAD(config)

    # Data
    datamodule = DataModule(config)
    # datamodule = AnomalyDatamodule(config)

    # Train model
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=logger,
        max_epochs=10_001,
        num_sanity_val_steps=2,
        callbacks=[
            ModelCheckpoint(str(ckpts_dir), monitor='train_loss', filename='TRAIN_{epoch:04d}_loss-{train_loss:.5f}', auto_insert_metric_name=False),
            ModelCheckpoint(str(ckpts_dir), monitor='val_loss', filename='VAL_{epoch:04d}_loss-{val_loss:.5f}', auto_insert_metric_name=False),
            ModelCheckpoint(str(ckpts_dir), monitor='val_auroc', filename='VAL_{epoch:04d}_auroc-{val_auroc:.5f}', auto_insert_metric_name=False, mode='max'),
            ModelCheckpoint(str(ckpts_dir), monitor='val_miou', filename='VAL_{epoch:04d}_miou-{val_miou:.5f}', auto_insert_metric_name=False, mode='max'),
        ]
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("-sde", "--sde", required=True)
    p.add_argument("-dataset", "--dataset", type=str, required=True)
    p.add_argument("-cat", "--category", type=str, required=True)
    cmd_line_args, _ = p.parse_known_args()

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    main(cmd_line_args)
