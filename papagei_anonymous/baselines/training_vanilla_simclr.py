# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os 
import torch 
import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
import wandb
import augmentations
import joblib
import torch_optimizer as toptim
from dataset import PPGDatasetVanillaSimCLR, generate_dataloader
from tqdm import tqdm
from models.transformer import TransformerSimple
from models import efficientnet
from models.resnet import ResNet1D
from pytorch_metric_learning import losses
from training import train_step, training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from functools import lru_cache
from torchvision import transforms
from training_pospair import harmonize_datasets
from torch_ecg._preprocessors import Normalize
from training_distributed import ddp_setup, save_model

torch.autograd.set_detect_anomaly(True)

def train_step(epoch, model, dataloader, criterion, optimizer, device):

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
    dataloader.sampler.set_epoch(epoch)

    signal_v1, signal_v2 = next(iter(dataloader))
    signal_v1, signal_v2 = signal_v1.to(device), signal_v2.to(device)

    embeddings_v1, _ = model(signal_v1)
    embeddings_v2, _ = model(signal_v2)
    loss = criterion(embeddings_v1, embeddings_v2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def training(model, epochs, train_dataloader, criterion, optimizer, device, directory, filename, wandb=None):

    """
    Training a SimCLR model

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
                                criterion=criterion,
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
            save_model(model, directory, filename, content, prefix="../")

        if device == "cuda:0" and step == epochs - 1:
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content, prefix="../") 

    return dict_log

def main(rank, world_size, epochs, batch_size):
    ddp_setup(rank, world_size)
    
    shuffle = True
    distributed = True
    lr = 0.0001
    prob_dictionary = {'g_p': 0.35, 'n_p': 0.20, 'w_p':0.0, 'f_p':0.20, 's_p':0.4, 'c_p':0.5}
    fs_target = 125

    simclr_transform = augmentations.get_transformations(g_p=prob_dictionary['g_p'],
                                            n_p=prob_dictionary['n_p'],
                                            w_p=prob_dictionary['w_p'],
                                            f_p=prob_dictionary['f_p'],
                                            s_p=prob_dictionary['s_p'],
                                            c_p=prob_dictionary['c_p']) 
    train_transform = transforms.Compose(simclr_transform)

    df = harmonize_datasets(prefix="../")

    dataset = PPGDatasetVanillaSimCLR(df=df,
                                fs_target=fs_target, 
                                transform=train_transform)

    sampler = DistributedSampler(dataset, shuffle=shuffle)
    train_dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    sampler=sampler,
                    drop_last=True)

    # model_config = {'d_model': 1250,
    #            'nhead': 2,
    #            'dim_feedforward': 2048,
    #            'trans_dropout': 0.0,
    #            'proj_dropout': 0.0,
    #            'num_layers': 2,
    #            'h1': 1024,
    #            'embedding_size': 512}
    # model = TransformerSimple(model_config=model_config)

    model_config = {'base_filters': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'groups': 1,
                    'n_block': 18,
                    'n_classes': 512,
                    }

    model = ResNet1D(in_channels=1, 
                base_filters=model_config['base_filters'], 
                kernel_size=model_config['kernel_size'],
                stride=model_config['stride'],
                groups=model_config['groups'],
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = "cuda:" + str(rank) 
    print(device)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion = losses.SelfSupervisedLoss(losses.NTXentLoss())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    ### Experiment Tracking ###
    experiment_name = "resnet"
    name = "vanilla_simclr"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size,
         "augmentations": prob_dictionary}

    # wandb.init(project=experiment_name,
    #         config=config | model_config, 
    #         name=name,
    #         group=group_name)

    # run_id = wandb.run.id
    wandb = None
    run_id = "a12d"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   criterion=criterion,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   filename=model_filename,
                   wandb=wandb)
    # wandb.finish()
    joblib.dump(dict_log, f"../models/{time}/{model_filename}_log.p")
    
    destroy_process_group()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    world_size = 8
    epochs = 20000
    batch_size = 128
    mp.spawn(main, args=(world_size, epochs, batch_size), nprocs=world_size)
