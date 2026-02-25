# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os 
import torch 
import sys
import pandas as pd
import torch.multiprocessing as mp
import wandb
import augmentations
import joblib
import torch_optimizer as toptim
from dataset import dataset_selector, PPGDataset
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
torch.autograd.set_detect_anomaly(True)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def save_model(model, directory, filename, content, prefix=""):
    """
    Helper function to save model.
    
    Args:
        model (torch.nn.Module): Model to train
        directory (string): directory to save model
        filename (string): model name for saving
        prefix (String): Prefix for correct path
    """
    # Check if directory exists, if not, create it
    if not os.path.exists(f"{prefix}../models/{directory}"):
        os.makedirs(f"{prefix}../models/{directory}")
    
    torch.save(model.state_dict(), f"{prefix}../models/{directory}/{filename}_{content}.pt")


def train_step(epoch, model, dataloader, criterion, optimizer, device):

    """
    One training step in distributed setting.

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
    train_loss = 0
    for i, (X, y) in enumerate(dataloader):
        print(f"[Rank {torch.distributed.get_rank()}] Processing batch {i}")
        signal_view1 = X[0].to(device)
        signal_view2 = X[1].to(device)

        z_1, _ = model(signal_view1)
        z_2, _ = model(signal_view2)
        loss = criterion(z_1, z_2)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)

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
        directory (string): directory to save model
        filename (string): model name for saving
        wandb (wandb): wandb object for experiment tracking

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    best_lost = float('inf')
    
    for e in tqdm(range(epochs)):
        epoch_loss = train_step(epoch=e,
                               model=model,
                               dataloader=train_dataloader,
                               criterion=criterion,
                               optimizer=optimizer,
                               device=device)

        if wandb and device=="cuda:0":
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"[{device}] Epoch: {e+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        if device == "cuda:0" and epoch_loss < best_lost:
            best_lost = epoch_loss
            print(f"Saving model to: {directory}")
            content = f"epoch{e+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

    return dict_log

def main(rank, world_size, dataset, epochs, batch_size):
    ddp_setup(rank, world_size)
    
    num_workers = 0
    shuffle = True
    distributed = True
    lr = 0.0001
    label_name = "age"
    prob_dictionary = {'g_p': 0.35, 'n_p': 0.20, 'w_p':0.0, 'f_p':0.20, 's_p':0.4, 'c_p':0.5}
    fs_target = 125
    use_projection = True

    simclr_transform = augmentations.get_transformations(g_p=prob_dictionary['g_p'],
                                            n_p=prob_dictionary['n_p'],
                                            w_p=prob_dictionary['w_p'],
                                            f_p=prob_dictionary['f_p'],
                                            s_p=prob_dictionary['s_p'],
                                            c_p=prob_dictionary['c_p']) 

    train_dataloader, val_dataloader, test_dataloader = dataset_selector(key=dataset,
                                                                        CustomDataset=PPGDataset,
                                                                        label_name=label_name,
                                                                        fs_target=fs_target,
                                                                        simclr_transform=simclr_transform,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle, 
                                                                        distributed=distributed)
    
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
                n_classes=model_config['n_classes'],
                use_projection=use_projection)

    # model_config = {'h1': 64,
    #                 'h2': 32,
    #                 'h3': 128,
    #                 'h4': 256,
    #                 'h5': 384,
    #                 'h6': 512,
    #                 'h7': 768,
    #                 'h8': 1024}

    # model = efficientnet.EfficientNetB0Base(in_channels=1, dict_channels=model_config)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = "cuda:" + str(rank) 
    print(device)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion = losses.SelfSupervisedLoss(losses.NTXentLoss())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    ### Experiment Tracking ###
    experiment_name = f"{dataset}_resnet"
    name = "papagei_p_emb"
    group_name = f"{dataset}_PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size,
         "augmentations": prob_dictionary}

    wandb.init(project=experiment_name,
            config=config | model_config, 
            name=name,
            group=group_name, 
            mode="offline")

    run_id = wandb.run.id
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
    wandb.finish()
    joblib.dump(dict_log, f"../models/{time}/{model_filename}_log.p")
    
    destroy_process_group()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    world_size = 8
    epochs = 15000
    batch_size = 128
    datasets = ['vital_mesa_mimic']
    for d in datasets:
        mp.spawn(main, args=(world_size, d, epochs, batch_size), nprocs=world_size)