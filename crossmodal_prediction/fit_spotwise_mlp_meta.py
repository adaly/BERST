import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lazy_datasets import VisiumSpotDataset, ArrayBatchSampler
from lazy_datasets import get_dataset_dimensions, compute_gene_statistics

from baseline_models import ImageMLPEncoder
from training import train_expression_model, run_epoch
from training import evaluate_per_gene, plot_training_history


################################  SPECIFY SPATIAL MODEL  ################################

import argparse
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-b', '--mlp-blocks', type=int, default=1, required=False,
                    help='Number of blocks in MLP expression encoder')
parser.add_argument('-m', '--metadata', type=str, default="none", required=False,
                    help='Level of metadata to use -- "tissue", "all", or "none"')
args = parser.parse_args()

n_blocks = args.mlp_blocks
use_meta = args.metadata

if use_meta not in ["none", "all", "tissue"]:
    raise ValueError('metadata flag must be set to "none", "tissue", or "all"')

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

output_dir = Path('/commons/groups/phatnani_lab/adaly/BERST/results/spotwise_mlp_meta')


#################################  LOAD DATA AND SPLITS  ################################

data_dir = Path('/commons/groups/phatnani_lab/adaly/BERST_data')
root_dir = data_dir / 'processed'
split = pickle.load(open(root_dir / 'TVTSPLIT_20260922.pickle', 'rb'))

train = split['train']
val = split['val']
test = split['test']

print(len(train), 'training arrays')
print(len(val), 'validation arrays')
print(len(test), 'test arrays')

assert len(np.intersect1d(train, val)) == 0
assert len(np.intersect1d(train, test)) == 0

if use_meta != "none":
    with open(root_dir / "metadata_encoding.json") as f:
        metadata_encoding = json.load(f)

    # How many values does each piece of categorical metadata have?
    # (used for constructing embedding)
    categorical_cardinalities = [
        len(metadata_encoding['categorical_encodings'][col])
        for col in metadata_encoding['categorical_columns']
    ]

    tissue_index = metadata_encoding['tissue_index']
    n_continuous = len(metadata_encoding['continuous_columns'])


############################  INSTANTIATE DATASETS & LOADERS  ###########################

train_dataset = VisiumSpotDataset(root_dir, array_ids=train, return_metadata=True)
val_dataset = VisiumSpotDataset(root_dir, array_ids=val, return_metadata=True)
test_dataset = VisiumSpotDataset(root_dir, array_ids=test, return_metadata=True)

# Determine input and output dimensions
input_dim, n_genes = get_dataset_dimensions(
    train_dataset
)

# Calculate input scaling parameters from train fold
gene_mean, gene_std = compute_gene_statistics(
    train_dataset
)

print(f"Image feature dimension: {input_dim}")
print(f"Number of target genes: {n_genes}")


train_sampler = ArrayBatchSampler(
    dataset=train_dataset,
    batch_size=512,
    shuffle_arrays=True,
    shuffle_spots=True,
    drop_last=False,
    seed=42,
)

val_sampler = ArrayBatchSampler(
    dataset=val_dataset,
    batch_size=1024,
    shuffle_arrays=False,
    shuffle_spots=False,
    drop_last=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)


##################################  INSTANTIATE MODEL  ##################################

# Initialize model
kwargs = {'metadata_mode':use_meta}
if use_meta != "none":
    kwargs['categorical_cardinalities'] = categorical_cardinalities
    kwargs['tissue_index'] = tissue_index
    if use_meta == "all":
        kwargs['n_continuous'] = n_continuous
    
mlp_model = ImageMLPEncoder(
    input_dim=input_dim,
    n_genes=n_genes,
    hidden_dim=1024,
    latent_dim=256,
    n_blocks=n_blocks,
    dropout=0.2,
    **kwargs,
)


#####################################  TRAIN MODEL  #####################################

mlp_model, mlp_history = train_expression_model(
    model=mlp_model,
    train_loader=train_loader,
    val_loader=val_loader,
    gene_mean=gene_mean,
    gene_std=gene_std,
    device=DEVICE,
    n_epochs=100,
    learning_rate=1e-4,
    weight_decay=1e-4,
    ridge_alpha=0.0,
    loss_type="mse",
    patience=10,
    checkpoint_path=output_dir/ Path("baseline1_mlp_b%d_%s.pt" % (n_blocks, use_meta)),
)

ax = plot_training_history({'MLP': mlp_history})

plt.savefig(output_dir / Path('baseline1_mlp_b%d_%s.png' % (n_blocks, use_meta)), bbox_inches='tight')


#####################################  TEST MODEL  ######################################

test_sampler = ArrayBatchSampler(
    dataset=test_dataset,
    batch_size=1024,
    shuffle_arrays=False,
    shuffle_spots=False,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)

mlp_test = run_epoch(
    model=mlp_model,
    dataloader=test_loader,
    gene_mean=gene_mean,
    gene_std=gene_std,
    device=DEVICE,
    optimizer=None,
)

print("MLP:", use_meta, 'metadata')
print(mlp_test)


##################################  PER-GENE METRICS  ###################################

mlp_gene_metrics = evaluate_per_gene(
    mlp_model,
    test_loader,
    gene_mean,
    gene_std,
    DEVICE,
)

mlp_gene_metrics.to_csv(output_dir / Path('baseline1_mlp_b%d_%s_genemetrics.csv' % (n_blocks, use_meta)), index=False)
