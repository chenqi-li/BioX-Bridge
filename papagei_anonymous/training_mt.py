# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pandas as pd 
import numpy as np
import os 
import torch 
import sys
sys.path.append("../papagei-foundation-model/")
import augmentations
import joblib
import torch.multiprocessing as mp
import wandb
import joblib
import torch_optimizer as toptim
from models.transformer import TransformerSimple
from models import efficientnet
from models.resnet import ResNet1DMoE
from pytorch_metric_learning import losses, miners 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
torch.autograd.set_detect_anomaly(True)
from dataset import PPGDatasetLabelsArray, generate_dataloader
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from training_distributed import ddp_setup, save_model
from training_pospair import harmonize_datasets

def train_step(epoch, model, dataloader, criterion1, criterion2, optimizer, device, miner=None, use_sqi=True, use_sqi_loss=False, alpha=0.8):
    
    """
    One training step for PaPaGei-S

    Args:
        epoch (int): Current step
        model (torch.nn.Module): Model to train
        dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion1 (torch.nn.<Loss>): Contrastive loss function
        criterion2 (torch.nn.<Loss>): Regression loss function
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        miner (pytorch metric learning miner): Use a hard sample mining method
        use_sqi (boolean): To use signal quality index for mining
        use_sqi_loss (boolean): Multi-task loss uses SQI in addition to contrastive and ipa
        alpha (float): a value between 0 and 1 to decide the contribution of losses

    Returns:
        loss (float): The training loss for the step
    """
    
    model.to(device)
    model.train()
    dataloader.sampler.set_epoch(epoch)

    X, y = next(iter(dataloader))
    signal, svri, sqi, ipa = X.to(device), y[:, 0].to(device), y[:, 1].to(device), y[:, 2].to(device)

    embeddings, ipa_pred, sqi_pred, _ = model(signal)

    # Use a miner?
    if miner:
        # Compute hard pairs using quality or svri?
        if use_sqi:
            hard_pairs = miner(embeddings, sqi)
        else:
            hard_pairs = miner(embeddings, svri)
        contrastive_loss = criterion1(embeddings, svri, hard_pairs)
    else:
        contrastive_loss = criterion1(embeddings, svri)
    # Predict raw IPA values
    ipa_loss = criterion2(ipa_pred, ipa.unsqueeze(dim=-1))

    if use_sqi_loss:
        sqi_loss = criterion2(sqi_pred, sqi.unsqueeze(dim=-1))
        loss = alpha * contrastive_loss + (1 - alpha)/2 * ipa_loss + (1 - alpha)/2 * sqi_loss
    else:
        loss = alpha * contrastive_loss + (1 - alpha) * ipa_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def training(model, epochs, train_dataloader, criterion1, criterion2, optimizer, device, directory, filename, miner=None, wandb=None, use_sqi=True, use_sqi_loss=False, alpha=0.8):

    """
    Training PaPaGei-S

    Args:
        model (torch.nn.Module): Model to train
        epochs (int): No. of epochs to train
        train_dataloader (torch.utils.data.Dataloader): A training dataloader with signals
        criterion1 (torch.nn.<Loss>): Contrastive loss function
        criterion2 (torch.nn.<Loss>): Regression loss function
        optimizer (torch.optim): Optimizer to modify weights
        device (string): training device; use GPU
        directory (string): directory to save model
        filename (string): model name for saving
        miner (pytorch metric learning miner): Use a hard sample mining method
        wandb (wandb): wandb object for experiment tracking
        use_sqi (boolean): To use signal quality index for mining
        use_sqi_loss (boolean): Multi-task loss uses SQI in addition to contrastive and ipa
        alpha (float): a value between 0 and 1 to decide the contribution of losses

    Returns:
        dict_log (dictionary): A dictionary log with metrics
    """

    dict_log = {'train_loss': []}
    best_loss = float('inf')
    
    for step in tqdm(range(epochs), desc="Training Progress"):
        epoch_loss = train_step(epoch=step,
                                model=model,
                                dataloader=train_dataloader,
                                criterion1=criterion1,
                                criterion2=criterion2,
                                optimizer=optimizer,
                                device=device,
                                miner=miner,
                                use_sqi=use_sqi,
                                alpha=alpha
                                )

        if wandb and device == "cuda:0":
            wandb.log({"Train Loss": epoch_loss})

        dict_log['train_loss'].append(epoch_loss)
        print(f"[{device}] Step: {step+1}/{epochs} | Train Loss: {epoch_loss:.4f}")

        if device == "cuda:0" and epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

        if device == "cuda:0" and step == epochs - 1:
            print(f"Saving model to: {directory}")
            content = f"step{step+1}_loss{epoch_loss:.4f}"
            save_model(model, directory, filename, content)

    return dict_log

def main(rank, world_size, epochs, batch_size):
    ddp_setup(rank, world_size)
    
    shuffle = True
    distributed = True
    lr = 0.0001
    prob_dictionary = {'g_p': 0.0, 'n_p': 0.0, 'w_p':0.0, 'f_p':0.0, 's_p':0.0, 'c_p':0.25}
    fs_target = 125
    bins_svri = 8
    bins_skewness = 0
    binary_ipa = False
    use_sqi = False
    use_sqi_loss = True
    alpha = 0.6
    dataset_name = "mesa_mimic"

    simclr_transform = augmentations.get_transformations(g_p=prob_dictionary['g_p'],
                                            n_p=prob_dictionary['n_p'],
                                            w_p=prob_dictionary['w_p'],
                                            f_p=prob_dictionary['f_p'],
                                            s_p=prob_dictionary['s_p'],
                                            c_p=prob_dictionary['c_p']) 
    train_transform = transforms.Compose(simclr_transform)

    df = harmonize_datasets(dataset_name=dataset_name)
    print(len(df))

    dataset = PPGDatasetLabelsArray(df=df,
                                fs_target=fs_target, 
                                transform=train_transform,
                                bins_svri=bins_svri,
                                bins_skewness=bins_skewness,
                                binary_ipa=binary_ipa)

    sampler = DistributedSampler(dataset, shuffle=shuffle)
    train_dataloader = DataLoader(dataset=dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    sampler=sampler,
                    drop_last=True)

    # model = ResNet1D(in_channels=1, 
    #             base_filters=model_config['base_filters'], 
    #             kernel_size=model_config['kernel_size'],
    #             stride=model_config['stride'],
    #             groups=model_config['groups'],
    #             n_block=model_config['n_block'],
    #             n_classes=model_config['n_classes'],
    #             use_mt_regression=True)

    model_config = {'base_filters': 32,
                    'kernel_size': 3,
                    'stride': 2,
                    'groups': 1,
                    'n_block': 18,
                    'n_classes': 512,
                    'n_experts': 3
                    }

    model = ResNet1DMoE(in_channels=1, 
                base_filters=model_config['base_filters'], 
                kernel_size=model_config['kernel_size'],
                stride=model_config['stride'],
                groups=model_config['groups'],
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'],
                n_experts=model_config['n_experts'])

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    device = "cuda:" + str(rank) 
    print(device)
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion1 = losses.NTXentLoss()
    criterion2 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # miner = miners.MultiSimilarityMiner()
    miner = None
    ### Experiment Tracking ###
    experiment_name = "resnet"
    name = f"mt_moe_{str(model_config['n_block'])}_{dataset_name}_"
    group_name = "PPG"

    config = {"learning_rate": lr, 
         "epochs": epochs,
         "batch_size": batch_size,
         "augmentations": prob_dictionary,
         "bins_svri": bins_svri,
         "bins_skewness": bins_skewness,
         "binary_ipa": binary_ipa,
         "use_sqi":use_sqi,
         "alpha":alpha}

    # wandb.init(project=experiment_name,
    #         config=config | model_config, 
    #         name=name,
    #         group=group_name)

    # run_id = wandb.run.id
    run_id= "kwdjiu"
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_filename = f'{experiment_name}_{name}_{run_id}_{time}'

    dict_log = training(model=model, 
                   train_dataloader=train_dataloader,
                   epochs=epochs,
                   criterion1=criterion1,
                   criterion2=criterion2,
                   optimizer=optimizer,
                   device=device,
                   directory=time,
                   filename=model_filename,
                   miner=miner,
                   wandb=None,
                   use_sqi=use_sqi,
                   alpha=alpha,
                   use_sqi_loss=use_sqi_loss)
    # wandb.finish()
    joblib.dump(dict_log, f"../models/{time}/{model_filename}_log.p")
    
    destroy_process_group()

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    world_size = 8
    epochs = 10000
    batch_size = 128
    mp.spawn(main, args=(world_size, epochs, batch_size), nprocs=world_size)