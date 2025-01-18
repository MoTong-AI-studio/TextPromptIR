import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from utils.dataset_utils import PromptTrainDataset
from net.model import TextPromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
class TextPromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = TextPromptIR()
        self.loss_fn  = nn.L1Loss()
    
    def forward(self, x, embedding):
        return self.net(x, embedding)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch,embedding) = batch
        restored = self.net(degrad_patch,embedding)

        loss = self.loss_fn(restored, clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)

        return [optimizer], [scheduler]






def main():
    print("Options")
    print(opt)
    os.environ["WANDB_API_KEY"] = "d207dd2982d61a5fc1efa1846e0b662af2dd84c4"
    wandb.login()
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="TextPromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")


    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir, every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,drop_last=True, num_workers=opt.num_workers)
    model = TextPromptIRModel()
    
    # trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, strategy="ddp_find_unused_parameters_true", logger=logger, callbacks=[checkpoint_callback])
    trainer = pl.Trainer(max_epochs=opt.epochs, accelerator="gpu", devices=opt.num_gpus, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()




