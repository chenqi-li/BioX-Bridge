import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
sys.path.append("../../../papagei-foundation-model/")
from transforms import DataTransform_FD, DataTransform_TD
from tfc_utils import NTXentLoss_poly, TFCDataset
import torch.fft as fft
import torch.nn as nn
import torch.optim as optim
import augmentations
import joblib
import torch
import wandb
import random
import os 
from tqdm import tqdm
from training_pospair import harmonize_datasets
from transforms import DataTransform_FD, DataTransform_TD
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from functools import lru_cache
from torch_ecg._preprocessors import Normalize
from models.resnet import BasicBlock, MyConv1dPadSame, MyMaxPool1dPadSame, TFCResNet
from training_distributed import save_model
from datetime import datetime

def train_step(epoch, model, dataloader, batch_size, optimizer, device):

    """
    One training epoch for a SimCLR model

    Args:
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU

    Returns:
        train_loss (float): The training loss for the epoch
    """

    model.to(device)
    model.train()

    data, aug1, data_f, aug1_f = next(iter(dataloader))
    data, aug1 = data.float().to(device), aug1.float().to(device)
    data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)

    """Produce embeddings"""
    h_t, z_t, h_f, z_f = model(data, data_f)
    h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

    """Compute Pre-train loss"""
    """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
    nt_xent_criterion = NTXentLoss_poly(device, batch_size, 0.2, True)

    loss_t = nt_xent_criterion(h_t, h_t_aug)
    loss_f = nt_xent_criterion(h_f, h_f_aug)
    l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss


    l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
    loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

    lam = 0.2
    loss = lam*(loss_t + loss_f) + l_TF

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def training(model, epochs, train_dataloader, batch_size, optimizer, device, directory, filename, wandb=None):

    """
    Training a TFC model

    Args:
        model (torch.nn.Module): Model to train
        epochs (int): No. of epochs to train
        train_dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion (torch.nn.<Loss>): Loss function to optimizer
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        wandb (wandb): wandb object for experiment tracking

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    best_loss = float('inf')
    
    for step in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = train_step(epoch=step,
                                model=model,
                                dataloader=train_dataloader,
                                batch_size=batch_size,
                                optimizer=optimizer,
                                device=device)

        if wandb and device == "cuda:0":
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"[{device}] Step: {step+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        if device == "cuda:0" and epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content, prefix="../../")

        if device == "cuda:0" and step == epochs - 1:
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content, prefix="../../") 

    return dict_log

def main(epochs, batch_size):
    shuffle = True
    lr = 0.0001
    fs_target = 125
    print("Loading datasets...")
    df = harmonize_datasets(prefix="../../")

    config = {'jitter_ratio':0.1}
    dataset = TFCDataset(df=df,
                        fs_target=fs_target,
                        config=config)
    
    train_dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    shuffle=shuffle,
                    drop_last=True)
    
    model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                }

    model = TFCResNet(model_config=model_config)
    device = "cuda:0"
    print(device)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

     ### Experiment Tracking ###
    experiment_name = "resnet"
    name = "tfc"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size}

    # wandb.init(project=experiment_name,
    #         config=config | model_config, 
    #         name=name,
    #         group=group_name)
    wandb = None
    # run_id = wandb.run.id
    run_id = "afgh"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   batch_size=batch_size,
                   filename=model_filename,
                   wandb=wandb)
    # wandb.finish()
    joblib.dump(dict_log, f"../../../models/{time}/{model_filename}_log.p")
    
if __name__ == "__main__":
    epochs = 15000
    batch_size = 256
    main(epochs=epochs, batch_size=batch_size)