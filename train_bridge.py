import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import seaborn as sn
from copy import deepcopy
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
import torch.utils
from LaBraM_anonymous.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import LaBraM_anonymous.modeling_finetune
from LaBraM_anonymous.engine_for_finetuning import evaluate
from utils import evaluate_bridge, fm_forward, create_forward_pre_hook, fm_input_hook, fm_output_hook, fm_lastlayer_output_hook
from utils import labram_data_prepare, hubertecg_data_prepare, hubertecgb_data_prepare, hubertecgl_data_prepare, ecgfm_data_prepare, papagei_data_prepare, ecgdualnet_data_prepare
from utils import labram_forward, hubertecg_forward, hubertecgb_forward, hubertecgl_forward, ecgfm_forward, papagei_forward, ecgdualnet_forward
from LaBraM_anonymous.utils import NativeScalerWithGradNormCount as NativeScaler
import LaBraM_anonymous.utils
from HuBERT_ECG_anonymous.code.hubert_ecg import HuBERTECG
from HuBERT_ECG_anonymous.code.hubert_ecg import HuBERTECGConfig
from HuBERT_ECG_anonymous.code.hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification
from papagei_anonymous.models.resnet import ResNet1DMoE
from papagei_anonymous.linearprobing.utils import load_model_without_module_prefix
from ECG_Classification_anonymous.ecg_classification.config import ECGAttNet_CONFIG_XL
from ECG_Classification_anonymous.ecg_classification.model import ECGAttNet
from NormWear_anonymous.main_model import NormWearModel
import utils
from scipy import interpolate
import shutil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import math
import datetime
from torch.utils.tensorboard import SummaryWriter
import mlflow
import sys
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import re
from utils import CorrelationLoss, plot_confusion_matrix, plot_bridge_error_distribution
from collections import defaultdict

loss_functions = {
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
}

def get_args():
    parser = argparse.ArgumentParser('Bridge Training', add_help=False)
    ##############
    ####LaBraM####
    ##############
    parser.add_argument('--experiment_number', type=int, help='Experiment number')
    parser.add_argument('--experiment_name', type=str, default='wesad_ppg2ecg', help='Experiment name for mlflow')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_dir', default='/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes', help='path where to save, empty for no saving')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--evaluate_every', default=5, type=int)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--fm_old', type=str, choices=['normwear', 'labram', 'hubertecg', 'hubertecgb', 'hubertecgl', 'ecgfm', 'papagei', 'ecgdualnet'], help='What model to use for fm_old')
    parser.add_argument('--fm_old_checkpoint', type=str, help='Checkpoint to load for fm_old')
    parser.add_argument('--fm_new', type=str, choices=['normwear', 'labram', 'hubertecg', 'hubertecgb', 'hubertecgl', 'ecgfm', 'papagei', 'ecgdualnet'], help='What model to use for fm_new')
    parser.add_argument('--fm_new_checkpoint', type=str, help='Checkpoint to load for fm_new')
    parser.add_argument('--ecg_channel_expansion', type=str, default=None, help='How to expand from 1 channel to 12 channels for ECG')
    parser.add_argument('--bridge_model', type=str, choices=['protoLRFC'])
    parser.add_argument('--bridge_rank', type=int)
    parser.add_argument('--bridge_criterion', type=str, choices=['L1Loss', 'MSELoss', 'CosineEmbeddingLoss'])
    parser.add_argument('--bridge_criterion_location', type=str, choices=['bridge_output', 'fm_old_lastlayer_output'], help='Where to calculate the bridge loss')
    parser.add_argument('--bridge_output_location', type=str, help='Where to connect the output of the bridge to the FM')
    parser.add_argument('--bridge_input_location', type=str, help='Where to connect the input of the bridge to the FM')
    parser.add_argument('--bridge_input_dim_reduction', type=str, help='How to reduce the dimension of the fm_new feature before feeding to the bridge')
    parser.add_argument('--bridge_proto_init', type=str, help="protoLRFC ONLY: How to initialize the prototypes")
    parser.add_argument('--bridge_sampler', default='sequential', type=str, choices=['sequential', 'random'], help='How to sample the data for the bridge training')
    parser.add_argument('--linear_probe_fm_new_unfreeze_fm', action='store_true', help='linear_probe_fm_new ONLY: Unfreeze the FM and train with the linear prober together')
    parser.add_argument('--linear_probe_fm_new_architecture', type=str, help='linear_probe_fm_new ONLY: what architecture to use for linear probing, e.g. one layer or two layer MLP')
    parser.add_argument('--linear_probe_fm_new_weighted_loss', type=json.loads, help='linear_probe_fm_new ONLY: List of weight for loss to use for each class, should give for all classes')
    parser.add_argument('--linear_probe_input_dim_reduction', type=str, help='How to reduce dimension of FM output before feeding into the linear prober')
    parser.add_argument('--train_size', type=float, default=100, help='Percentage of training data to use')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'train', 'linear_probe_fm_new', 'store_features'])
    parser.add_argument('--debug_run', action='store_true', help='Break training and evaluation loops early to go through the entire code before running')
    return parser.parse_args()

# mlflow Logging
def log_mlflow(args, stats, split, step):
    for metric in args.metrics:
        mlflow.log_metric(f"{metric}/{split}", stats[metric], step=step)
    if args.mode in ['train', 'evaluate']:
        mlflow.log_metric(f"loss_logits/{split}", stats['loss_logits'], step=step)
        mlflow.log_metric(f"loss_bridge/{split}", stats['loss_bridge'], step=step)
    elif args.mode in ['linear_probe_fm_new', 'check_before_train']:
        mlflow.log_metric(f"loss_logits/{split}", stats['loss_logits'], step=step)
    mlflow.log_figure(plot_confusion_matrix(stats['cm_percent'], stats['cm_percent'].index, title='Confusion Matrix (Percentage)'), f"cm_percent/{split}_epoch_{str(step).zfill(4)}.png")
    mlflow.log_figure(plot_confusion_matrix(stats['cm_abs'], stats['cm_abs'].index, title='Confusion Matrix (Absolute)'), f"cm_abs/{split}_epoch_{str(step).zfill(4)}.png")
    return None

# Loading LaBraM checkpoint
def load_labram(args, checkpoint_path):
    if 'FOG' in args.data_dir:
        fm = create_model(
            'labram_base_patch200_200_16',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=False,
            use_abs_pos_emb=False, # disable because our input 30s for ISRUC is longer than pretrained model 5s
            init_values=0.1,
            qkv_bias=False,
        ) # make sure these are the same as the configuration when it was trained
    else:
        fm = create_model(
            'labram_base_patch200_200',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=False,
            use_abs_pos_emb=False, # disable because our input 30s for ISRUC is longer than pretrained model 5s
            init_values=0.1,
            qkv_bias=False,
        ) # make sure these are the same as the configuration when it was trained
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Load ckpt from %s" % checkpoint_path)
    checkpoint_model = None
    for model_key in 'model|module'.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    # For removing the "student." prefix and use the model for downstream tasks after pretraining, see modeling_finetune.py NeuralTransformer vs modeling_pretrain.py NeuralTransformerForMaskedEEGModeling
    if checkpoint_path.endswith('labram-base.pth'):
        if (checkpoint_model is not None): #and (args.model_filter_name != '') we remove this as default
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                    print(f"Removing prefix student. in model key")
                else:
                    pass
            checkpoint_model = new_dict
    state_dict = fm.state_dict()
    # Remove the classifier, if the model classifier and checkpoint classifier sizes do not match
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    all_keys = list(checkpoint_model.keys())
    # Relative positional encoding appears to be used in older version of LaBraM, remove it
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
            print(f"Removing key {key} from pretrained checkpoint")
    # Load the model
    utils.load_state_dict(fm, checkpoint_model, prefix='')
    fm.eval()
    for name, param in fm.named_parameters():
        param.requires_grad = False # freeze all layer
    return fm # Note: we do not use the head of the model and will be using our own linear_probe head for consistency with other models

# Loading HuBERTECG checkpoint
def load_hubertecg(args, checkpoint_path):
    # Note: checkpoint is downloaded from hugging face via wget
    print(f"Loading pretrained model from {checkpoint_path.split('/')[-1]}")
    # Using older version of torch, which uses nn.utils.weight_norm instead of nn.utils.parametrizations. They name the weight_norm parameters differently, so we need to replace the key original0=weight_g and original1=weight_v. Just replacing key is enough, see nn.utils.parametrizations.weight_norm source code for torch 2.5
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    renamed_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.endswith('conv.parametrizations.weight.original0'):
            new_key = key.replace('conv.parametrizations.weight.original0', 'conv.weight_g')
        elif key.endswith('conv.parametrizations.weight.original1'):
            new_key = key.replace('conv.parametrizations.weight.original1', 'conv.weight_v')
        else:
            new_key = key
        renamed_state_dict[new_key] = value
    config = checkpoint["model_config"]
    fm = HuBERTECG(config)
    load_result = fm.load_state_dict(renamed_state_dict, strict=False)
    # Report missing and unexpected keys
    if load_result.missing_keys:
        print("Missing keys:")
        for key in load_result.missing_keys:
            print(f"  {key}")
    if load_result.unexpected_keys:
        print("Unexpected keys:")
        for key in load_result.unexpected_keys:
            print(f"  {key}")    
    return fm
load_hubertecgb = load_hubertecgl = load_hubertecg # share same loader for all models of different sizes

# Loading ECG-FM checkpoint
def load_ecgfm(args, checkpoint_path):
    # # Download pretrained checkpoint, you can copy paste the following in a python session from terminal
    # from huggingface_hub import hf_hub_download
    # _ = hf_hub_download(
    #     repo_id='wanglab/ecg-fm-preprint',
    #     filename='mimic_iv_ecg_physionet_pretrained.pt',
    #     local_dir=os.path.join('/users/anonymous/BioX-Bridge/ECG_FM_anonymous', 'ckpts'),
    # )
    # _ = hf_hub_download(
    #     repo_id='wanglab/ecg-fm-preprint',
    #     filename='mimic_iv_ecg_physionet_pretrained.yaml',
    #     local_dir=os.path.join('/users/anonymous/BioX-Bridge/ECG_FM_anonymous', 'ckpts'),
    # )

    # Load model
    from fairseq_signals.models import build_model_from_checkpoint

    fm = build_model_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    return fm

# Loading PaPaGei checkpoint
def load_papagei(args, checkpoint_path):
    model_config = {'base_filters': 32,
                'kernel_size': 3,
                'stride': 2,
                'groups': 1,
                'n_block': 18,
                'n_classes': 512,
                'n_experts': 3
                }

    fm = ResNet1DMoE(in_channels=1, 
                base_filters=model_config['base_filters'], 
                kernel_size=model_config['kernel_size'],
                stride=model_config['stride'],
                groups=model_config['groups'],
                n_block=model_config['n_block'],
                n_classes=model_config['n_classes'],
                n_experts=model_config['n_experts'])

    model = load_model_without_module_prefix(fm, checkpoint_path)
    return model

def load_ecgdualnet(args, checkpoint_path):
    config = ECGAttNet_CONFIG_XL
    config["classes"] = 2
    network = ECGAttNet(config=config)
    state_dict = torch.load(checkpoint_path)
    model_state_dict = network.state_dict()
    state_dict = {key: value for key, value in state_dict.items() if model_state_dict[key].shape == value.shape}
    model_state_dict.update(state_dict)
    load_result = network.load_state_dict(model_state_dict)
    print(load_result)
    for name, module in network.named_children():
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {param_count:,} parameters")
    return network

def load_normwear(args, checkpoint_path):
    model = NormWearModel(weight_path=checkpoint_path, optimized_cwt=True)
    return model

# Define the prototype bridge
class protoLRFC(nn.Module):
    def __init__(self, in_features, out_features, r, num_prototypes, num_tokens):
        super(protoLRFC, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.P = num_prototypes
        self.num_tokens = num_tokens
        
        if r > 0:
            self.A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.B = nn.Parameter(torch.zeros(self.P * self.num_tokens, self.r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        
        self.prototypes = nn.Parameter(torch.randn(self.out_features, self.P))
    
    def initialize_prototypes(self, selected_prototypes):
        selected_prototypes_tensor = torch.from_numpy(selected_prototypes).to(
            dtype=self.prototypes.dtype,
            device=self.prototypes.device
        )
        self.prototypes.data = selected_prototypes_tensor

    def forward(self, x):
        curr_bs = x.size(0)
        if self.r > 0:
            attn_flat = x @ self.A.transpose(0,1) @ self.B.transpose(0,1) # (self.P*self.num_tokens)
            attn = attn_flat.view(curr_bs, self.P, self.num_tokens) # (self.P, self.num_tokens)
            output = self.prototypes @ attn # (self.out_features, self.num_tokens)
        return output

def main(args):
    # Assert the data directory and foundation models are compatible
    parts = args.data_dir.split("_") # Split data_dir by "_"
    pattern = re.compile(r"(.*?fm)(\d+|[A-Za-z]+)$")  # Captures "XXXfm" + trailing value
    matches = []
    for part in parts:
        match = pattern.match(part)
        if match:
            prefix, fm_value = match.groups()
            matches.append(fm_value)
    expected_values = {v if v not in {'hubertecgl', 'hubertecgb'} else 'hubertecg' for v in [args.fm_new, args.fm_old]} #remove the b and l postfix for hubertecg
    found_values = set(matches)
    assert found_values == expected_values, f"Dataset and foundation model does not match! Foundation models: {expected_values}, Dataset for: {found_values}"

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Define device
    device = torch.device('cuda')

    # Print arguments
    print(f'Experiment arguments: {args}')

    # mlflow Logger
    mlflow.set_experiment(args.experiment_name)
    mlflow.start_run(run_name=f"{str(args.experiment_number).zfill(3)}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    mlflow.log_params(vars(args)) # log all the hyperparameters
    if args.debug_run:
        mlflow.set_tag("debug_run", "True")

    # Get dataset and define the associated variables
    train_files = os.listdir(os.path.join(args.data_dir, "train"))
    val_files = os.listdir(os.path.join(args.data_dir, "val"))
    test_files = os.listdir(os.path.join(args.data_dir, "test"))
    pair_files = os.listdir(os.path.join(args.data_dir, "pair"))
    if 'ISRUC' in args.data_dir:
        if args.fm_old in ['labram'] and args.fm_new in ['ecgfm', 'hubertecg', 'hubertecgb', 'hubertecgl', 'ecgdualnet']:
            data_order = 'eegecglabel'
        elif args.fm_old in ['ecgfm', 'hubertecg', 'hubertecgb', 'hubertecgl', 'ecgdualnet'] and args.fm_new in ['labram']:
            data_order = 'ecgeeglabel'
        else:
            raise ValueError("Invalid FM combination")
        train_dataset = utils.ISRUCLoader(os.path.join(args.data_dir, "train"), train_files, data_order)
        val_dataset = utils.ISRUCLoader(os.path.join(args.data_dir, "val"), val_files, data_order)
        test_dataset = utils.ISRUCLoader(os.path.join(args.data_dir, "test"), test_files, data_order)
        pair_dataset = utils.ISRUCLoader(os.path.join(args.data_dir, "pair"), pair_files, data_order, percentage=args.train_size)
    elif 'WESAD' in args.data_dir:
        if args.fm_old in ['papagei'] and args.fm_new in ['ecgfm', 'hubertecg', 'hubertecgb', 'hubertecgl', 'ecgdualnet']:
            data_order = 'ppgecglabel'
        elif args.fm_old in ['ecgfm', 'hubertecg', 'hubertecgb', 'hubertecgl', 'ecgdualnet'] and args.fm_new in ['papagei']:
            data_order = 'ecgppglabel'
        else:
            raise ValueError("Invalid FM combination")
        train_dataset = utils.WESADLoader(os.path.join(args.data_dir, "train"), train_files, data_order)
        val_dataset = utils.WESADLoader(os.path.join(args.data_dir, "val"), val_files, data_order)
        test_dataset = utils.WESADLoader(os.path.join(args.data_dir, "test"), test_files, data_order)
        pair_dataset = utils.WESADLoader(os.path.join(args.data_dir, "pair"), pair_files, data_order, percentage=args.train_size)
    elif 'FOG' in args.data_dir:
        if args.fm_old in ['labram'] and args.fm_new in ['normwear']:
            data_order = 'eegemglabel'
        elif args.fm_old in ['normwear'] and args.fm_new in ['labram']:
            data_order = 'emgeeglabel'
        else:
            raise ValueError("Invalid FM combination")
        train_dataset = utils.FOGLoader(os.path.join(args.data_dir, "train"), train_files, data_order)
        val_dataset = utils.FOGLoader(os.path.join(args.data_dir, "val"), val_files, data_order)
        test_dataset = utils.FOGLoader(os.path.join(args.data_dir, "test"), test_files, data_order)
        pair_dataset = utils.FOGLoader(os.path.join(args.data_dir, "pair"), pair_files, data_order, percentage=args.train_size)
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)
    if args.bridge_sampler in ['sequential']:
        sampler_pair = torch.utils.data.SequentialSampler(pair_dataset)
    elif args.bridge_sampler in ['random']:
        sampler_pair = torch.utils.data.RandomSampler(pair_dataset)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=int(args.batch_size),
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=int(args.batch_size),
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
            test_dataset, sampler=sampler_test,
            batch_size=int(args.batch_size),
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
    data_loader_pair = torch.utils.data.DataLoader(
            pair_dataset, sampler=sampler_pair,
            batch_size=int(args.batch_size),
            num_workers=2,
            pin_memory=True,
            drop_last=False
        )
    args.nb_classes = int(re.search(r'(\d+)classes', args.data_dir).group(1))
    # Define metrics
    args.metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro"]
    
    # Load FM from old modality
    print(f'Loading FM from old modality')
    fm_old = globals()[f"load_{args.fm_old}"](args=args, checkpoint_path=args.fm_old_checkpoint)
    fm_old.to(device)
    print(f"FM from old modality loaded, with architecture: {str(fm_old)}")

    # Load FM from new modality
    print(f'Loading FM from new modality')
    fm_new = globals()[f"load_{args.fm_new}"](args=args, checkpoint_path=args.fm_new_checkpoint)
    fm_new.to(device)
    print(f"FM from new modality loaded, with architecture: {str(fm_new)}")

    # Load linear prober weights for FM from old modality
    if args.mode in ['train', 'evaluate', 'store_features']:
        prober_checkpoint_path = re.sub(r"(\d{3})\.pth$", r"\1_linear_prober.pth", args.fm_old_checkpoint)
        prober_checkpoint = torch.load(prober_checkpoint_path)
        prober_size = prober_checkpoint['linear_prober_state_dict']['weight'].shape
        linear_prober = nn.Linear(prober_size[1], prober_size[0]).to(device)
        linear_prober.load_state_dict(prober_checkpoint['linear_prober_state_dict'])
        linear_prober.eval()
        print(f"Linear prober loaded, with size {prober_size}")
    
    # Setup the bridge
    if args.mode in ['train', 'evaluate', 'store_features']:
        # Define the bridge input dimensions
        if args.fm_new in ['hubertecg']:
            if any(loc in args.bridge_input_location for loc in ['encoder.layers', 'feature_extractor']):
                if args.bridge_input_dim_reduction in ['mean']:
                    args.bridge_input_dim = 512
                elif args.bridge_input_dim_reduction is None:
                    args.bridge_input_dim = 93*512*6
        elif args.fm_new in ['hubertecgb']:
            if any(loc in args.bridge_input_location for loc in ['encoder.layers', 'feature_extractor']):
                if args.bridge_input_dim_reduction in ['mean']:
                    args.bridge_input_dim = 768
                elif args.bridge_input_dim_reduction is None:
                    args.bridge_input_dim = 93*768*6
        elif args.fm_new in ['hubertecgl']:
            if any(loc in args.bridge_input_location for loc in ['encoder.layers', 'feature_extractor']):
                if args.bridge_input_dim_reduction in ['mean']:
                    args.bridge_input_dim = 960
                elif args.bridge_input_dim_reduction is None:
                    args.bridge_input_dim = 93*960*6
        elif args.fm_new in ['ecgfm']:
            if any(loc in args.bridge_input_location for loc in ['encoder.layers']):
                if args.bridge_input_dim_reduction in ['mean']:
                    args.bridge_input_dim = 768
                elif args.bridge_input_dim_reduction is None:
                    args.bridge_input_dim = 768*1875
            elif any(loc in args.bridge_input_location for loc in ['feature_extractor']):
                if args.bridge_input_dim_reduction in ['mean']:
                    args.bridge_input_dim = 256
                elif args.bridge_input_dim_reduction is None:
                    args.bridge_input_dim = 256*1875
        elif args.fm_new in ['ecgdualnet']:
            if args.bridge_input_dim_reduction in ['mean']:
                args.layer_to_size_map = {
                    'spectrogram_encoder[0]': 128,
                    'spectrogram_encoder[1]': 256,
                    'spectrogram_encoder[2]': 512,
                    'spectrogram_encoder[3]': 512,
                    'spectrogram_encoder[4]': 256
                }
                args.bridge_input_dim = args.layer_to_size_map[args.bridge_input_location]
        elif args.fm_new in ['papagei']:
            args.layer_to_size_map = {
                'basicblock_list[0]': 32,
                'basicblock_list[1]': 32,
                'basicblock_list[2]': 32,
                'basicblock_list[3]': 32,
                'basicblock_list[4]': 64,
                'basicblock_list[5]': 64,
                'basicblock_list[6]': 64,
                'basicblock_list[7]': 64,
                'basicblock_list[8]': 128,
                'basicblock_list[9]': 128,
                'basicblock_list[10]': 128,
                'basicblock_list[11]': 128,
                'basicblock_list[12]': 256,
                'basicblock_list[13]': 256,
                'basicblock_list[14]': 256,
                'basicblock_list[15]': 256,
                'basicblock_list[16]': 512,
                'basicblock_list[17]': 512
            }
            args.bridge_input_dim = args.layer_to_size_map[args.bridge_input_location]
        elif args.fm_new in ['labram']:
            if any(loc in args.bridge_input_location for loc in ['blocks']):
                 if args.bridge_input_dim_reduction in ['mean']:
                     args.bridge_input_dim = 200
        elif args.fm_new in ['normwear']:
            if any(loc in args.bridge_input_location for loc in ['encoder_blocks']):
                 if args.bridge_input_dim_reduction in ['mean']:
                     args.bridge_input_dim = 768
        # Define the bridge output dimensions
        if args.fm_old in ['labram']:
            if 'blocks' in args.bridge_output_location:
                if 'ISRUC' in args.data_dir:
                    args.bridge_output_dim = 181*200
                elif 'FOG' in args.data_dir:
                    args.bridge_output_dim = 19*200
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 200
        elif args.fm_old in ['ecgfm']:
            if any(loc in args.bridge_output_location for loc in ['encoder.layers']):
                if 'ISRUC' in args.data_dir:
                    args.bridge_output_dim = 768*937
                elif 'WESAD' in args.data_dir:
                    args.bridge_output_dim = 768*1875
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 768
            elif any(loc in args.bridge_output_location for loc in ['feature_extractor']):
                args.bridge_output_dim = 256*1875
        elif args.fm_old in ['hubertecg']:
            if any(loc in args.bridge_output_location for loc in ['encoder.layers']):
                if 'ISRUC' in args.data_dir:
                    args.bridge_output_dim = 512*46
                elif 'WESAD' in args.data_dir:
                    args.bridge_output_dim = 512*93
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 512
        elif args.fm_old in ['hubertecgb']:
            if any(loc in args.bridge_output_location for loc in ['encoder.layers']):
                if 'ISRUC' in args.data_dir:
                    args.bridge_output_dim = 768*46
                elif 'WESAD' in args.data_dir:
                    args.bridge_output_dim = 768*93
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 768
        elif args.fm_old in ['hubertecgl']:
            if any(loc in args.bridge_output_location for loc in ['encoder.layers']):
                if 'ISRUC' in args.data_dir:
                    args.bridge_output_dim = 960*46
                elif 'WESAD' in args.data_dir:
                    args.bridge_output_dim = 960*93
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 960
        elif args.fm_old in ['ecgdualnet']:
            if any(loc in args.bridge_output_location for loc in ['encoder.layers']):
                if 'ISRUC' in args.data_dir:
                    args.bridge_output_dim = 256*40
                elif 'WESAD' in args.data_dir:
                    args.bridge_output_dim = 256*80
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 256
        elif args.fm_old in ['papagei']:
            args.layer_to_size_map = {
                                    'basicblock_list[0]': (32, 7500),
                                    'basicblock_list[1]': (32, 7500),
                                    'basicblock_list[2]': (32, 3750),
                                    'basicblock_list[3]': (32, 3750),
                                    'basicblock_list[4]': (32, 1875),
                                    'basicblock_list[5]': (64, 1875),
                                    'basicblock_list[6]': (64, 938),
                                    'basicblock_list[7]': (64, 938),
                                    'basicblock_list[8]': (64, 469),
                                    'basicblock_list[9]': (128, 469),
                                    'basicblock_list[10]': (128, 235),
                                    'basicblock_list[11]': (128, 235),
                                    'basicblock_list[12]': (128, 118),
                                    'basicblock_list[13]': (256, 118),
                                    'basicblock_list[14]': (256, 59),
                                    'basicblock_list[15]': (256, 59),
                                    'basicblock_list[16]': (256, 30),
                                    'basicblock_list[17]': (512, 30)
                                }
            if 'basicblock_list' in args.bridge_output_location:
                args.bridge_output_dim = math.prod(args.layer_to_size_map[args.bridge_output_location])
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 512
        elif args.fm_old in ['normwear']:
            if 'encoder_blocks' in args.bridge_output_location:
                if 'emgfmnormwear' in args.data_dir:
                    args.bridge_output_dim = 1680*768
            elif 'linear_prober' in args.bridge_output_location:
                args.bridge_output_dim = 768
        if args.fm_old in ['labram']:
            if 'ISRUC' in args.data_dir:
                args.out_features = 200
                args.num_tokens = 181
            elif 'FOG' in args.data_dir:
                args.out_features = 200
                args.num_tokens = 19
        elif args.fm_old in ['ecgfm']:
            args.out_features = 768
            if 'ISRUC' in args.data_dir:
                args.num_tokens = 937
            elif 'WESAD' in args.data_dir:
                args.num_tokens = 1875
        elif args.fm_old in ['hubertecg']: 
            args.out_features = 512
            if 'ISRUC' in args.data_dir:
                args.num_tokens = 46
            elif 'WESAD' in args.data_dir:
                args.num_tokens = 93
        elif args.fm_old in ['hubertecgb']: 
            args.out_features = 768
            if 'ISRUC' in args.data_dir:
                args.num_tokens = 46
            elif 'WESAD' in args.data_dir:
                args.num_tokens = 93
        elif args.fm_old in ['hubertecgl']: 
            args.out_features = 960
            if 'ISRUC' in args.data_dir:
                args.num_tokens = 46
            elif 'WESAD' in args.data_dir:
                args.num_tokens = 93
        elif args.fm_old in ['ecgdualnet']:
            args.out_features = 256
            if 'ISRUC' in args.data_dir:
                args.num_tokens = 40
            elif 'WESAD' in args.data_dir:
                args.num_tokens = 80
        elif args.fm_old in ['papagei']:
            args.out_features = args.layer_to_size_map[args.bridge_output_location][0]
            args.num_tokens = args.layer_to_size_map[args.bridge_output_location][1]
        elif args.fm_old in ['normwear']:
            if 'encoder_blocks' in args.bridge_output_location:
                if 'emgfmnormwear' in args.data_dir:
                    args.num_tokens = 1680
                    args.out_features = 768
            elif 'linear_prober' in args.bridge_output_location:
                args.num_tokens = 1
                args.out_features = 768

        # Initialize the bridge, its loss function, and its optimizer
        bridge_model = protoLRFC(in_features=args.bridge_input_dim, out_features=args.out_features, r=args.bridge_rank, num_prototypes=int(args.bridge_proto_init.split("_")[1]), num_tokens=args.num_tokens).to(device)
        bridge_criterion = loss_functions[args.bridge_criterion]() #nn.MSELoss()
        optimizer = optim.Adam(bridge_model.parameters(), lr=args.lr)
        n_parameters = sum(p.numel() for p in bridge_model.parameters() if p.requires_grad)
        print(f'Number of parameters in the bridge: {n_parameters:,}')

        # Load the bridge if evaluate
        if args.mode in ['evaluate']:
            checkpoint_path = os.path.join(args.output_dir, f'{str(args.epoch).zfill(3)}.pth')
            checkpoint_pt = torch.load(checkpoint_path)
            bridge_load_result = bridge_model.load_state_dict(checkpoint_pt['model_state_dict'])
            print('Bridge loading', bridge_load_result)

    # Define the linear probe input dim
    if (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['hubertecg']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['hubertecg']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 512
        elif args.linear_probe_input_dim_reduction is None:
            args.linear_probe_input_dim = 93*512
    elif (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['hubertecgb']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['hubertecgb']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 768
        elif args.linear_probe_input_dim_reduction is None:
            args.linear_probe_input_dim = 93*768
    elif (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['hubertecgl']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['hubertecgl']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 960
        elif args.linear_probe_input_dim_reduction is None:
            args.linear_probe_input_dim = 93*960
    elif (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['ecgfm']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['ecgfm']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 768
        elif args.linear_probe_input_dim_reduction is None:
            args.linear_probe_input_dim = 768*1875
    elif (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['ecgdualnet']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['ecgdualnet']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 256
        elif args.linear_probe_input_dim_reduction is None:
            args.linear_probe_input_dim = 80*256
    elif (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['papagei']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['papagei']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 512
        elif args.linear_probe_input_dim_reduction is None:
            args.linear_probe_input_dim = 512*15
    elif (args.mode in ['linear_probe_fm_new'] and args.fm_new in ['labram']) or (args.mode in ['train', 'evaluate', 'store_features'] and args.fm_old in ['labram']):
        if args.linear_probe_input_dim_reduction in ['mean']:
            args.linear_probe_input_dim = 200
    
    # Set fm to eval mode
    fm_old.eval()
    fm_new.eval()
    
    # Train the bridge
    if args.mode in ['train']:

        # Create hooks
        messenger = {}
        if 'linear_prober' in args.bridge_output_location:
            if args.fm_old in ['labram']:
                fm_input_hook_handle = fm_old.blocks[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['papagei']:
                fm_input_hook_handle = fm_old.basicblock_list[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['ecgfm']:
                fm_input_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
                fm_input_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['ecgdualnet']:
                fm_input_hook_handle = fm_old.spectrogram_encoder[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['normwear']:
                fm_input_hook_handle = fm_old.backbone.encoder_blocks[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
        elif 'linear_prober' not in args.bridge_output_location:
            fm_input_hook_handle = eval(f"fm_old.{args.bridge_output_location}").register_forward_hook(partial(fm_input_hook, messenger=messenger))
        fm_output_hook_handle = eval(f"fm_new.{args.bridge_input_location}").register_forward_hook(partial(fm_output_hook, messenger=messenger))
        if args.fm_old in ['labram']:
            fm_lastlayer_output_hook_handle = fm_old.blocks[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['papagei']:
            fm_lastlayer_output_hook_handle = fm_old.basicblock_list[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['ecgfm']:
            fm_lastlayer_output_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
            fm_lastlayer_output_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['ecgdualnet']:
            fm_lastlayer_output_hook_handle = fm_old.spectrogram_encoder[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['normwear']:
            fm_lastlayer_output_hook_handle = fm_old.backbone.encoder_blocks[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        
        # # Evaluate to make sure the finetuned FM and linear prober was saved and loaded properly
        # with torch.no_grad():
        #     args.mode = 'check_before_train'
        #     linear_prober.eval()
        #     stats = evaluate_bridge(args=args, data_loader=data_loader_train, bridge_criterion=nn.CrossEntropyLoss(), fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
        #     log_mlflow(args=args, stats=stats, split='train', step=-1)
        #     print(f"Train set: Accuracy {stats['accuracy']}")
        #     stats = evaluate_bridge(args=args, data_loader=data_loader_val, bridge_criterion=nn.CrossEntropyLoss(), fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
        #     log_mlflow(args=args, stats=stats, split='val', step=-1)
        #     print(f"Val set: Accuracy {stats['accuracy']}")
        #     stats = evaluate_bridge(args=args, data_loader=data_loader_test, bridge_criterion=nn.CrossEntropyLoss(), fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
        #     log_mlflow(args=args, stats=stats, split='test', step=-1)
        #     print(f"Test set: Accuracy {stats['accuracy']} Balanced Accuracy {stats['balanced_accuracy']} F1 Macro {stats['f1_macro']} F1 Weighted {stats['f1_weighted']}")
        #     args.mode = 'train'

        if args.bridge_proto_init.split("_")[0] in ['randreal']:
            shuffled_prototypes_path = os.path.join(args.data_dir, f"shuffled_prototypes_{args.fm_old}[{int(args.bridge_output_location.split('[')[1].split(']')[0])}].npy")
            
            # Save if shuffled_prototypes for the fm_old does not exist
            if not os.path.exists(shuffled_prototypes_path):
                fm_old_input_list = []
                for step, batch in enumerate(data_loader_pair):
                    # Prepare the data
                    current_batch_size = batch[0].shape[0]
                    data_old = batch[0].to(device)
                    data_new = batch[1].to(device)
                    target = batch[-1]
                    target = target.to(device, non_blocking=True)
                    data_new = eval(f"{args.fm_new}_data_prepare")(args=args, device=device, data=data_new)
                    data_old = eval(f"{args.fm_old}_data_prepare")(args=args, device=device, data=data_old)

                    # Forward for FMs
                    with torch.no_grad():
                        fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=fm_old, fm_new=fm_new, data_old=data_old, data_new=data_new, messenger=messenger, args=args)
                    
                    fm_old_input_list.append(fm_old_input)
                fm_old_input_all = torch.cat(fm_old_input_list, dim=0)
                assert fm_old_input.shape[-1] == args.out_features
                fm_old_input_all = fm_old_input_all.detach().cpu().numpy().reshape(-1, args.out_features)
                np.random.shuffle(fm_old_input_all)
                np.save(shuffled_prototypes_path, fm_old_input_all)

            # Load the shuffled prototypes
            shuffled_prototypes = np.load(shuffled_prototypes_path)
            bridge_model.initialize_prototypes(shuffled_prototypes[:int(args.bridge_proto_init.split("_")[1]),:].T)
            torch.cuda.empty_cache()

        # Iterate through the epochs
        for e in range(args.epoch):
            # Iterate through the dataloader
            for step, batch in enumerate(data_loader_pair):
                # Zero the gradients
                optimizer.zero_grad()

                # Prepare the data
                current_batch_size = batch[0].shape[0]
                data_old = batch[0].to(device)
                data_new = batch[1].to(device)
                target = batch[-1]
                target = target.to(device, non_blocking=True)
                data_new = eval(f"{args.fm_new}_data_prepare")(args=args, device=device, data=data_new)
                data_old = eval(f"{args.fm_old}_data_prepare")(args=args, device=device, data=data_old)

                # Forward for FMs
                with torch.no_grad():
                    fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=fm_old, fm_new=fm_new, data_old=data_old, data_new=data_new, messenger=messenger, args=args)

                # Compute bridge features from new features
                if args.bridge_input_dim_reduction in ['mean']:
                    if step == 0: # only calculate the dimension to reduce at the first step
                        dims_to_reduce = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != args.bridge_input_dim]
                    fm_new_output = torch.mean(fm_new_output, dim=tuple(dims_to_reduce))
                bridge_output = bridge_model(fm_new_output.view(current_batch_size,-1))

                # Change the features to the target shape of fm_old intermediate input
                if args.fm_old in ['labram']:
                    if 'ISRUC' in args.data_dir:
                        bridge_output = bridge_output.reshape(current_batch_size, 181, 200)
                    elif 'FOG' in args.data_dir:
                        bridge_output = bridge_output.reshape(current_batch_size, 19, 200)
                elif args.fm_old in ['papagei']:
                    pass
                elif args.fm_old in ['ecgfm']:
                    bridge_output = bridge_output.permute(2, 0, 1)
                elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
                    bridge_output = bridge_output.permute(0, 2, 1)
                elif args.fm_old in ['ecgdualnet']:
                    bridge_output = bridge_output.permute(2, 0, 1)
                elif args.fm_old in ['normwear']:
                    if any(loc in args.bridge_output_location for loc in ['encoder_blocks']):
                        if 'emgfmnormwear' in args.data_dir:
                            bridge_output = bridge_output.permute(0, 2, 1)
                            bridge_output = bridge_output.view(current_batch_size, 3, 560, 768)
                            bridge_output = bridge_output.reshape(current_batch_size * 3, 560, 768)
                    elif 'linear_prober' in args.bridge_output_location:
                        bridge_output = bridge_output.view(current_batch_size, 768)
                # # Optional debug
                # print(f'fm_old_input and fm_new_output collected')
                # print(f'fm_old_input: {fm_old_input.shape}, fm_new_output: {fm_new_output.shape}')

                # Calculate bridge loss
                if args.bridge_criterion_location in ['bridge_output']:
                    if args.bridge_criterion in ['CosineEmbeddingLoss']:
                        loss_bridge = bridge_criterion(bridge_output.view(current_batch_size, -1), fm_old_input.view(current_batch_size, -1), torch.ones(current_batch_size).to(device))
                    elif args.bridge_criterion in ['CosineEmbeddingLoss_tokenwise']:
                        bridge_output = bridge_output.view(-1, 200)
                        fm_old_input = fm_old_input.view(-1, 200)
                        loss_bridge = bridge_criterion(bridge_output, fm_old_input, torch.ones(fm_old_input.size(0)).to(device))
                    else:
                        loss_bridge = bridge_criterion(bridge_output.view(current_batch_size, -1), fm_old_input.view(current_batch_size, -1))
                elif args.bridge_criterion_location in ['fm_old_lastlayer_output']:    
                    # Send bridge_output to fm_old and get fm_old lastlayer feature
                    if 'linear_prober' not in args.bridge_output_location:
                        messenger['bridge_output'] = bridge_output
                        hook_handle = eval(f"fm_old.{args.bridge_output_location}").register_forward_pre_hook(
                            create_forward_pre_hook(messenger)
                        )
                        messenger = eval(f"{args.fm_old}_forward")(fm=fm_old, data=data_old, messenger=messenger)
                        fm_old_lastlayer_output_pred = messenger['fm_lastlayer_output']
                        hook_handle.remove()
                        # Reshape for normwear
                        if args.fm_old in ['normwear']:
                            BN, P, E = fm_old_lastlayer_output_pred.shape
                            fm_old_lastlayer_output_pred = torch.reshape(fm_old_lastlayer_output_pred,(data_old.shape[0], -1, P, E))
                    elif 'linear_prober' in args.bridge_output_location:
                        fm_old_lastlayer_output_pred = bridge_output
                        if step == 0: # only calculate the dimension to reduce at the first step
                            probe_dims_to_reduce = [i for i, dim in enumerate(fm_old_lastlayer_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                        fm_old_lastlayer_output = torch.mean(fm_old_lastlayer_output, dim=tuple(probe_dims_to_reduce))

                    # Calculate the loss
                    if args.bridge_criterion in ['CosineEmbeddingLoss']:
                        loss_bridge = bridge_criterion(fm_old_lastlayer_output_pred.view(current_batch_size, -1), fm_old_lastlayer_output.view(current_batch_size, -1), torch.ones(current_batch_size).to(device))
                    elif args.bridge_criterion in ['CosineEmbeddingLoss_tokenwise']:
                        fm_old_lastlayer_output_pred = fm_old_lastlayer_output_pred.view(-1, 200)
                        fm_old_lastlayer_output = fm_old_lastlayer_output.view(-1, 200)
                        loss_bridge = bridge_criterion(fm_old_lastlayer_output_pred, fm_old_lastlayer_output, torch.ones(fm_old_lastlayer_output.size(0)).to(device))
                    else:
                        loss_bridge = bridge_criterion(fm_old_lastlayer_output_pred.view(current_batch_size, -1), fm_old_lastlayer_output.view(current_batch_size, -1))

                
                # Backward pass
                loss_bridge.backward()
                optimizer.step()

                # Record loss
                mlflow.log_metric("loss_bridge_iter", loss_bridge.item(), step=e*len(data_loader_train)+step)

                # Optional testrun debug
                if args.debug_run and step == 2:
                    break
            
            # Evaluate to check bridge performance
            if (e+1) % args.evaluate_every == 0:
                with torch.no_grad():
                    bridge_model.eval()
                    stats = evaluate_bridge(args=args, data_loader=data_loader_train, bridge_criterion=bridge_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=bridge_model, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
                    log_mlflow(args=args, stats=stats, split='train', step=e)
                    print(f"Model at epoch {e} on train set: Loss_bridge {stats['loss_bridge']} Accuracy {stats['accuracy']}")
                    stats = evaluate_bridge(args=args, data_loader=data_loader_val, bridge_criterion=bridge_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=bridge_model, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
                    log_mlflow(args=args, stats=stats, split='val', step=e)
                    print(f"Model at epoch {e} on val set: Loss_bridge {stats['loss_bridge']} Accuracy {stats['accuracy']}")
                    stats = evaluate_bridge(args=args, data_loader=data_loader_test, bridge_criterion=bridge_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=bridge_model, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
                    log_mlflow(args=args, stats=stats, split='test', step=e)
                    print(f"Model at epoch {e} on test set: Loss_bridge {stats['loss_bridge']} Accuracy {stats['accuracy']}")
                        
                    print(f'Model trained for {e} epoch')
                    print(f'Bridge output target: {fm_old_input}')
                    print(f'Bridge output prediction: {bridge_output}')
                    print(f'Bridge input: {fm_new_output}')
                    print("\n" * 5)
                    bridge_model.train()
            
            # Save model and optimizer states
            if (e+1) % args.evaluate_every == 0:
                checkpoint_path = os.path.join(args.output_dir, f'{str(e).zfill(3)}.pth')
                torch.save({
                    'epoch': e,  # Current epoch
                    'model_state_dict': bridge_model.state_dict(),  # Model parameters
                    'optimizer_state_dict': optimizer.state_dict()  # Optimizer parameters
                }, checkpoint_path)

            # Optional testrun debug
            if args.debug_run and e == 10:
                break
    # Evaluate an existing checkpoint
    elif args.mode in ['evaluate']:
        # Create hooks
        messenger = {}
        if 'linear_prober' in args.bridge_output_location:
            if args.fm_old in ['labram']:
                fm_input_hook_handle = fm_old.blocks[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['papagei']:
                fm_input_hook_handle = fm_old.basicblock_list[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['ecgfm']:
                fm_input_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
                fm_input_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['ecgdualnet']:
                fm_input_hook_handle = fm_old.spectrogram_encoder[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
            elif args.fm_old in ['normwear']:
                fm_input_hook_handle = fm_old.backbone.encoder_blocks[-1].register_forward_hook(partial(fm_input_hook, messenger=messenger))
        elif 'linear_prober' not in args.bridge_output_location:
            fm_input_hook_handle = eval(f"fm_old.{args.bridge_output_location}").register_forward_hook(partial(fm_input_hook, messenger=messenger))
        fm_output_hook_handle = eval(f"fm_new.{args.bridge_input_location}").register_forward_hook(partial(fm_output_hook, messenger=messenger))
        if args.fm_old in ['labram']:
            fm_lastlayer_output_hook_handle = fm_old.blocks[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['papagei']:
            fm_lastlayer_output_hook_handle = fm_old.basicblock_list[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['ecgfm']:
            fm_lastlayer_output_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
            fm_lastlayer_output_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['ecgdualnet']:
            fm_lastlayer_output_hook_handle = fm_old.spectrogram_encoder[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['normwear']:
            fm_lastlayer_output_hook_handle = fm_old.backbone.encoder_blocks[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        
        # Evaluate the bridge
        with torch.no_grad():
            bridge_model.eval()
            # stats = evaluate_bridge(args=args, data_loader=data_loader_train, bridge_criterion=bridge_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=bridge_model, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
            # log_mlflow(args=args, stats=stats, split='train', step=args.epoch)
            # print(f"Model at epoch {args.epoch} on train set: Loss_bridge {stats['loss_bridge']} Accuracy {stats['accuracy']}")
            # stats = evaluate_bridge(args=args, data_loader=data_loader_val, bridge_criterion=bridge_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=bridge_model, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
            # log_mlflow(args=args, stats=stats, split='val', step=args.epoch)
            # print(f"Model at epoch {args.epoch} on val set: Loss_bridge {stats['loss_bridge']} Accuracy {stats['accuracy']}")
            stats = evaluate_bridge(args=args, data_loader=data_loader_test, bridge_criterion=bridge_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=bridge_model, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
            log_mlflow(args=args, stats=stats, split='test', step=args.epoch)
            print(f"Model at epoch {args.epoch} on test set: Loss_bridge {stats['loss_bridge']} Accuracy {stats['accuracy']}")
    # Linear probing for fm_new
    elif args.mode in ['linear_probe_fm_new']:
        # Create hooks
        messenger = {}
        if args.fm_new in ['labram']:
            fm_output_hook_handle = fm_new.blocks[-1].register_forward_hook(partial(fm_output_hook, messenger=messenger))
        elif args.fm_new in ['papagei']:
            fm_output_hook_handle = fm_new.basicblock_list[-1].register_forward_hook(partial(fm_output_hook, messenger=messenger))
        elif args.fm_new in ['ecgfm']:
            fm_output_hook_handle = fm_new.encoder.layers[-1].register_forward_hook(partial(fm_output_hook, messenger=messenger))
        elif args.fm_new in ['hubertecg', 'hubertecgb', 'hubertecgl']:
            fm_output_hook_handle = fm_new.encoder.layers[-1].register_forward_hook(partial(fm_output_hook, messenger=messenger))
        elif args.fm_new in ['ecgdualnet']:
            fm_output_hook_handle = fm_new.spectrogram_encoder[-1].register_forward_hook(partial(fm_output_hook, messenger=messenger))
        elif args.fm_new in ['normwear']:
            fm_output_hook_handle = fm_new.backbone.encoder_blocks[-1].register_forward_hook(partial(fm_output_hook, messenger=messenger))


        # Unfreeze the feature_extractor and transformer layers as necessary
        fm_new.requires_grad_(False)
        if args.linear_probe_fm_new_unfreeze_fm:
            if args.fm_new in ['hubertecg', 'hubertecgb', 'hubertecgl']:
                fm_new.encoder.requires_grad_(True)
            if args.fm_new in ['papagei']:
                fm_new.requires_grad_(True)
            if args.fm_new in ['ecgfm']:
                fm_new.requires_grad_(True)
            if args.fm_new in ['ecgdualnet']:
                fm_new.requires_grad_(True)
            if args.fm_new in ['labram']:
                fm_new.requires_grad_(True)
            if args.fm_new in ['normwear']:
                fm_new.requires_grad_(True)

        # Define the linear prober
        if args.linear_probe_fm_new_architecture in ['one_layer']:
            linear_prober = nn.Linear(args.linear_probe_input_dim, args.nb_classes).to(device)
        elif args.linear_probe_fm_new_architecture in ['two_layer']:
            linear_prober = nn.Sequential(
                nn.Linear(args.linear_probe_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, args.nb_classes)
            ).to(device)
        print(f'Linear prober architecture: {linear_prober}')
        
        # Define the optimizer
        if args.linear_probe_fm_new_unfreeze_fm:
            prober_optimizer = optim.Adam(
                [
                    {'params': fm_new.parameters()},  # Parameters of fm_new
                    {'params': linear_prober.parameters()}  # Parameters of linear_prober
                ],
                lr=args.lr
            )
        else:
            prober_optimizer = optim.Adam(linear_prober.parameters(), lr=args.lr)
        
        # Define the loss
        if args.linear_probe_fm_new_weighted_loss:
            prober_criterion = nn.CrossEntropyLoss(weight=torch.tensor(1/np.array(args.linear_probe_fm_new_weighted_loss),dtype=torch.float32)).to(device)
        else:
            prober_criterion = nn.CrossEntropyLoss()
        
        # Summary of the setup
        n_parameters = sum(p.numel() for p in fm_new.parameters() if p.requires_grad)
        n_parameters += sum(p.numel() for p in linear_prober.parameters() if p.requires_grad)
        print(f'Number of trainable linear probing parameters: {n_parameters:,}')
        
        # Iterate through the epochs
        for e in range(args.epoch):
            # Set model to train mode if unfreeze
            if args.linear_probe_fm_new_unfreeze_fm:
                fm_new.train()

            # Iterate through the dataloader
            for step, batch in enumerate(data_loader_train):
                messenger.clear()
                # Zero the gradients
                prober_optimizer.zero_grad()

                # Prepare the data
                current_batch_size = batch[0].shape[0]
                data_old = batch[0].to(device)
                data_new = batch[1].to(device)
                target = batch[-1]
                target = target.to(device, non_blocking=True)
                data_new = eval(f"{args.fm_new}_data_prepare")(args=args, device=device, data=data_new)
                data_old = eval(f"{args.fm_old}_data_prepare")(args=args, device=device, data=data_old)

                # Forward for FMs
                fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=None, fm_new=fm_new, data_old=None, data_new=data_new, messenger=messenger, args=args)
                if fm_new_output is None and args.linear_probe_fm_new_unfreeze_fm and args.fm_new == 'hubertecg': # hubertecg have layer drop during training, we skip these batches
                    continue
                # Prepare the features from FM for input to the linear prober
                if args.linear_probe_input_dim_reduction in ['mean']:    
                    if step == 0: # only calculate the dimension to reduce at the first step
                        probe_dims_to_reduce = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                    fm_new_output = torch.mean(fm_new_output, dim=tuple(probe_dims_to_reduce))
                elif args.linear_probe_input_dim_reduction is None:
                    fm_new_output = fm_new_output.view(current_batch_size, -1)
                
                # Forward with the linear prober
                class_prediction = linear_prober(fm_new_output)
                loss_probe = prober_criterion(class_prediction, target)
            
                # Backward pass
                loss_probe.backward()
                prober_optimizer.step()

                # Record loss
                mlflow.log_metric("loss_probe_iter", loss_probe.item(), step=e*len(data_loader_train)+step)

                # Optional testrun debug
                if args.debug_run and step == 2:
                    break
            
            # Evaluate to check bridge performance
            if (e+1) % args.evaluate_every == 0:
                fm_old.eval()
                fm_new.eval()
                with torch.no_grad():
                    linear_prober.eval()
                    stats = evaluate_bridge(args=args, data_loader=data_loader_train, bridge_criterion=prober_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
                    log_mlflow(args=args, stats=stats, split='train', step=e)
                    print(f"Model at epoch {e} on train set: Accuracy {stats['accuracy']}")
                    stats = evaluate_bridge(args=args, data_loader=data_loader_val, bridge_criterion=prober_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
                    log_mlflow(args=args, stats=stats, split='val', step=e)
                    print(f"Model at epoch {e} on val set: Accuracy {stats['accuracy']}")
                    stats = evaluate_bridge(args=args, data_loader=data_loader_test, bridge_criterion=prober_criterion, fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
                    log_mlflow(args=args, stats=stats, split='test', step=e)
                    print(f"Model at epoch {e} on test set: Accuracy {stats['accuracy']}")
                        
                    print(f'Model trained for {e} epoch')
                    print("\n" * 5)
                    linear_prober.train()
            
            # Save weights
            if (e+1) % args.evaluate_every == 0:
                # Save the FM weights (even if it is frozen) and linear prober weights
                checkpoint_path = os.path.join(args.output_dir, f'{str(e).zfill(3)}.pth')
                checkpoint_linear_prober_path = os.path.join(args.output_dir, f'{str(e).zfill(3)}_linear_prober.pth')
                # Save FM weights
                if args.fm_new in ['papagei']:
                    torch.save(fm_new.state_dict(), checkpoint_path)
                elif args.fm_new in ['ecgfm']:
                    # Save FM weights
                    checkpoint = torch.load(args.fm_new_checkpoint)
                    checkpoint['model'] = fm_new.state_dict()
                    torch.save(checkpoint, checkpoint_path)
                elif args.fm_new in ['hubertecg', 'hubertecgb', 'hubertecgl']:
                    # Save FM weights
                    checkpoint = torch.load(args.fm_new_checkpoint)
                    checkpoint['model_state_dict'] = fm_new.state_dict()
                    torch.save(checkpoint, checkpoint_path)
                elif args.fm_new in ['ecgdualnet']:
                    torch.save(fm_new.state_dict(), checkpoint_path)
                elif args.fm_new in ['labram']:
                    to_save = {
                        'model': fm_new.state_dict()
                        }
                    torch.save(to_save, checkpoint_path)
                elif args.fm_new in ['normwear']:
                    checkpoint = torch.load(args.fm_new_checkpoint)
                    checkpoint['model'] = fm_new.state_dict()
                    torch.save(checkpoint, checkpoint_path)

                # Save linear prober weights
                torch.save({
                    'epoch': e,  # Current epoch
                    'linear_prober_state_dict': linear_prober.state_dict(),  # Model parameters
                    'prober_optimizer_state_dict': prober_optimizer.state_dict()  # Optimizer parameters
                }, checkpoint_linear_prober_path)
            
            # Optional testrun debug
            if args.debug_run and e == 10:
                break
            
            # Close all plots
            plt.close('all')
    # Save intermediate feature for metric
    elif args.mode in ['store_features']:
        # Note args.bridge_output_location and args.bridge_input_location are dummy placeholders for this mode
        # Create hooks
        messenger = {}
        if args.fm_old in ['labram']:
            fm_lastlayer_output_hook_handle = fm_old.blocks[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['papagei']:
            fm_lastlayer_output_hook_handle = fm_old.basicblock_list[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['ecgfm']:
            fm_lastlayer_output_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['hubertecg', 'hubertecgb', 'hubertecgl']:
            fm_lastlayer_output_hook_handle = fm_old.encoder.layers[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['ecgdualnet']:
            fm_lastlayer_output_hook_handle = fm_old.spectrogram_encoder[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        elif args.fm_old in ['normwear']:
            fm_lastlayer_output_hook_handle = fm_old.backbone.encoder_blocks[-1].register_forward_hook(partial(fm_lastlayer_output_hook, messenger=messenger))
        # # Evaluate to make sure the finetuned FM and linear prober was saved and loaded properly
        # with torch.no_grad():
        #     args.mode = 'check_before_train'
        #     linear_prober.eval()
        #     stats = evaluate_bridge(args=args, data_loader=data_loader_train, bridge_criterion=nn.CrossEntropyLoss(), fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
        #     log_mlflow(args=args, stats=stats, split='train', step=-1)
        #     print(f"Train set: Accuracy {stats['accuracy']}")
        #     stats = evaluate_bridge(args=args, data_loader=data_loader_val, bridge_criterion=nn.CrossEntropyLoss(), fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
        #     log_mlflow(args=args, stats=stats, split='val', step=-1)
        #     print(f"Val set: Accuracy {stats['accuracy']}")
        #     stats = evaluate_bridge(args=args, data_loader=data_loader_test, bridge_criterion=nn.CrossEntropyLoss(), fm_old=fm_old, fm_new=fm_new, bridge_model=None, messenger=messenger, device=device, ch_names=None, metrics=args.metrics, is_binary=(args.nb_classes == 1), linear_prober=linear_prober)
        #     log_mlflow(args=args, stats=stats, split='test', step=-1)
        #     print(f"Test set: Accuracy {stats['accuracy']}")
        #     args.mode = 'train'
        #     fm_input_hook_handle.remove()
        #     fm_output_hook_handle.remove()

        # Dictionary for FM configurations
        fm_config = {
            'labram': ['blocks', 12], #12
            'papagei': ['basicblock_list', 18], #18
            'ecgfm': ['encoder.layers', 12], #12
            'hubertecg': ['encoder.layers', 8], #8
            'hubertecgb': ['encoder.layers', 12], #8
            'hubertecgl': ['encoder.layers', 16], #8
            'ecgdualnet': ['spectrogram_encoder', 5], #4
            'normwear': ['backbone.encoder_blocks', 12] #12
        }
        
        # Define layer to bridge input and output size map for papagei
        if args.fm_new in ['papagei']:
            layer_to_bridge_input_size_map = {
                'basicblock_list[0]': 32,
                'basicblock_list[1]': 32,
                'basicblock_list[2]': 32,
                'basicblock_list[3]': 32,
                'basicblock_list[4]': 64,
                'basicblock_list[5]': 64,
                'basicblock_list[6]': 64,
                'basicblock_list[7]': 64,
                'basicblock_list[8]': 128,
                'basicblock_list[9]': 128,
                'basicblock_list[10]': 128,
                'basicblock_list[11]': 128,
                'basicblock_list[12]': 256,
                'basicblock_list[13]': 256,
                'basicblock_list[14]': 256,
                'basicblock_list[15]': 256,
                'basicblock_list[16]': 512,
                'basicblock_list[17]': 512
            }
        elif args.fm_new in ['ecgdualnet']:
            layer_to_bridge_input_size_map = {
                'spectrogram_encoder[0]': 128,
                'spectrogram_encoder[1]': 256,
                'spectrogram_encoder[2]': 512,
                'spectrogram_encoder[3]': 512,
                'spectrogram_encoder[4]': 256
            }
        if args.fm_old in ['papagei']:
            layer_to_bridge_output_size_map = {
                'basicblock_list[0]': (32, 7500),
                'basicblock_list[1]': (32, 7500),
                'basicblock_list[2]': (32, 3750),
                'basicblock_list[3]': (32, 3750),
                'basicblock_list[4]': (32, 1875),
                'basicblock_list[5]': (64, 1875),
                'basicblock_list[6]': (64, 938),
                'basicblock_list[7]': (64, 938),
                'basicblock_list[8]': (64, 469),
                'basicblock_list[9]': (128, 469),
                'basicblock_list[10]': (128, 235),
                'basicblock_list[11]': (128, 235),
                'basicblock_list[12]': (128, 118),
                'basicblock_list[13]': (256, 118),
                'basicblock_list[14]': (256, 59),
                'basicblock_list[15]': (256, 59),
                'basicblock_list[16]': (256, 30),
                'basicblock_list[17]': (512, 30)
            }
        elif args.fm_old in ['ecgdualnet']:
            layer_to_bridge_output_size_map = {
                'spectrogram_encoder[0]': (128, 281, 16),
                'spectrogram_encoder[1]': (256, 140, 8),
                'spectrogram_encoder[2]': (512, 70, 4),
                'spectrogram_encoder[3]': (512, 35, 2),
                'spectrogram_encoder[4]': (256, 17, 1)
            }

        # Iterate through the dataloader
        bridge_input_features = defaultdict(list)
        bridge_output_features = defaultdict(list)
        pseudo_label = defaultdict(list)

        for step, batch in enumerate(data_loader_pair):
            # Zero the gradients
            optimizer.zero_grad()

            # Prepare the data
            current_batch_size = batch[0].shape[0]
            data_old = batch[0].to(device)
            data_new = batch[1].to(device)
            target = batch[-1]
            target = target.to(device, non_blocking=True)
            data_new = eval(f"{args.fm_new}_data_prepare")(args=args, device=device, data=data_new)
            data_old = eval(f"{args.fm_old}_data_prepare")(args=args, device=device, data=data_old)

            # Loop through bridge_inputs
            for fm_layer in range(max(fm_config[args.fm_old][1], fm_config[args.fm_new][1])):
                if fm_layer <= fm_config[args.fm_old][1]-1:
                    fm_input_hook_handle = eval(f"fm_old.{fm_config[args.fm_old][0]}[{fm_layer}]").register_forward_hook(partial(fm_input_hook, messenger=messenger))
                    fm_old_append = True
                elif fm_layer > fm_config[args.fm_old][1]-1:
                    fm_input_hook_handle = eval(f"fm_old.{fm_config[args.fm_old][0]}[-1]").register_forward_hook(partial(fm_input_hook, messenger=messenger))
                    fm_old_append = False
                if fm_layer <= fm_config[args.fm_new][1]-1:
                    fm_output_hook_handle = eval(f"fm_new.{fm_config[args.fm_new][0]}[{fm_layer}]").register_forward_hook(partial(fm_output_hook, messenger=messenger))
                    fm_new_append = True
                elif fm_layer > fm_config[args.fm_new][1]-1:
                    fm_output_hook_handle = eval(f"fm_new.{fm_config[args.fm_new][0]}[-1]").register_forward_hook(partial(fm_output_hook, messenger=messenger))
                    fm_new_append = False
                
                # Forward for FMs
                with torch.no_grad():
                    fm_old_input, fm_new_output, fm_old_lastlayer_output = fm_forward(fm_old=fm_old, fm_new=fm_new, data_old=data_old, data_new=data_new, messenger=messenger, args=args)
                
                # Prepare the features from FM for input to the linear prober
                if args.linear_probe_input_dim_reduction in ['mean']:    
                    if step == 0: # only calculate the dimension to reduce at the first step
                        probe_dims_to_reduce = [i for i, dim in enumerate(fm_old_lastlayer_output.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                    fm_old_lastlayer_output = torch.mean(fm_old_lastlayer_output, dim=tuple(probe_dims_to_reduce))
                elif args.linear_probe_input_dim_reduction is None:
                    fm_old_lastlayer_output = fm_old_lastlayer_output.view(current_batch_size, -1)
                
                # Forward with the linear prober
                class_prediction = linear_prober(fm_old_lastlayer_output)
                
                # Reduce the fm_old feature as well
                if step == 0 and fm_layer == 0: # only calculate the dimension to reduce at the first step
                    assert len(fm_old_input.shape) == len(set(fm_old_input.shape)), f"fm_old_input shape dimensions are not unique: {fm_old_input.shape}"
                    assert len(fm_new_output.shape) == len(set(fm_new_output.shape)), f"fm_new_output shape dimensions are not unique: {fm_new_output.shape}"

                    if args.fm_old in ['papagei']:
                        dims_to_reduce_old = [i for i, dim in enumerate(fm_old_input.shape) if dim != current_batch_size and dim != layer_to_bridge_output_size_map[f'basicblock_list[{fm_layer}]'][0]]
                    elif args.fm_old in ['ecgdualnet']:
                        dims_to_reduce_old = [i for i, dim in enumerate(fm_old_input.shape) if dim != current_batch_size and dim != layer_to_bridge_output_size_map[f'spectrogram_encoder[{fm_layer}]'][0]]
                    elif args.fm_old in ['normwear', 'labram', 'ecgfm', 'hubertecg', 'hubertecgb', 'hubertecgl']:
                        dims_to_reduce_old = [i for i, dim in enumerate(fm_old_input.shape) if dim != current_batch_size and dim != args.linear_probe_input_dim]
                    if args.fm_new in ['papagei']:
                        dims_to_reduce_new = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != layer_to_bridge_input_size_map[f'basicblock_list[{fm_layer}]']]
                    elif args.fm_new in ['ecgdualnet']:
                        dims_to_reduce_new = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != layer_to_bridge_input_size_map[f'spectrogram_encoder[{fm_layer}]']]
                    elif args.fm_new in ['normwear', 'labram', 'ecgfm', 'hubertecg', 'hubertecgb', 'hubertecgl']:
                        dims_to_reduce_new = [i for i, dim in enumerate(fm_new_output.shape) if dim != current_batch_size and dim != args.bridge_input_dim]
                fm_old_input = torch.mean(fm_old_input, dim=tuple(dims_to_reduce_old))
                if args.fm_new in ['ecgfm', 'ecgdualnet', 'normwear']:
                    fm_new_output = torch.mean(fm_new_output, dim=tuple(dims_to_reduce_new))
                assert len(fm_old_input.shape) == 2
                assert fm_new_output.shape[0] == current_batch_size

                if fm_new_append:
                    bridge_input_features[fm_layer].append(fm_new_output.cpu().detach())
                if fm_old_append:
                    bridge_output_features[fm_layer].append(fm_old_input.cpu().detach().reshape(current_batch_size, -1))
                pseudo_label[fm_layer].append(class_prediction.cpu().detach().reshape(current_batch_size, -1))

                # Remove hook handle
                fm_input_hook_handle.remove()
                fm_output_hook_handle.remove()
            if args.debug_run and step == 2:
                break
        
        # Save tensor
        print('Saving bridge_input_features')
        for key, tensor_list in bridge_input_features.items():
            concatenated = torch.cat(tensor_list, dim=0)
            np.save(os.path.join(args.output_dir, f'bridge_input_features_{str(key).zfill(2)}.npy'), concatenated.numpy())
            print(key, concatenated.shape)
        print('Saving bridge_output_features')
        for key, tensor_list in bridge_output_features.items():
            concatenated = torch.cat(tensor_list, dim=0)
            np.save(os.path.join(args.output_dir, f'bridge_output_features_{str(key).zfill(2)}.npy'), concatenated.numpy())
            print(key, concatenated.shape)
        print('Saving pseudo_label')
        for key, tensor_list in pseudo_label.items():
            concatenated = torch.cat(tensor_list, dim=0)
            np.save(os.path.join(args.output_dir, f'pseudo_label_{str(key).zfill(2)}.npy'), concatenated.numpy())
            print(key, concatenated.shape)
    
    # End mlflow logger
    mlflow.end_run()

if __name__ == '__main__':
    # Read arguments
    opts = get_args()
    if opts.experiment_number:
        opts.output_dir = f'/data/anonymous/BioX-Bridge/checkpoints_{opts.experiment_name}/experiment_{str(opts.experiment_number).zfill(3)}'
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        opts.log_dir = f'/users/anonymous/BioX-Bridge/checkpoints_{opts.experiment_name}/experiment_{str(opts.experiment_number).zfill(3)}'
        Path(opts.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize the print_output.txt and redirect all prints to the text file, will be restored to stdout later
    if opts.mode in ['evaluate']:
        output_file_path = os.path.join(opts.log_dir, f"print_output_{opts.mode}.txt")
    elif opts.mode in ['train', 'linear_probe_fm_new', 'store_features']:
        output_file_path = os.path.join(opts.log_dir, f"print_output_{opts.mode}.txt")
    sys.stdout = open(output_file_path, 'w', buffering=1)
    
    # Track start time
    starttime = datetime.datetime.now()
    print(f'Start time: {starttime}')
    
    # Run the program
    main(opts)

    # Print peak vram usage
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak VRAM usage: {peak:.2f} GB")

    # Track end time
    endtime = datetime.datetime.now()
    print(f'End time: {endtime}')
    print(f'Experiment run duration: {str(endtime-starttime)}')

    # Restore original stdout at the end of the script
    sys.stdout.close()
    sys.stdout = sys.__stdout__