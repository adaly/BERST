import os
import sys
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

from training import train_expression_model, run_epoch
from training import evaluate_per_gene, plot_training_history


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

output_dir = Path('/commons/groups/phatnani_lab/adaly/BERST/results/spotwise_ridge')

alpha = float(sys.argv[1])
alpha_str = ('%.1e' % alpha).replace('.0', '')  # filename-friendly string, e.g 1e-04

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


############################  INSTANTIATE DATASETS & LOADERS  ###########################

train_dataset = VisiumSpotDataset(root_dir, array_ids=train)
val_dataset = VisiumSpotDataset(root_dir, array_ids=val)
test_dataset = VisiumSpotDataset(root_dir, array_ids=test)

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

class LinearExpressionModel(nn.Module):

    def __init__(
        self,
        input_dim,
        n_genes,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            n_genes,
        )

    def forward(self, x):

        return self.linear(x)

# Initialize model
linear_model = LinearExpressionModel(
    input_dim=input_dim,
    n_genes=n_genes,
)


#####################################  TRAIN MODEL  #####################################

linear_model, linear_history = train_expression_model(
    model=linear_model,
    train_loader=train_loader,
    val_loader=val_loader,
    gene_mean=gene_mean,
    gene_std=gene_std,
    device=DEVICE,
    n_epochs=50,
    learning_rate=1e-3,
    weight_decay=0.0,
    ridge_alpha=1e-4,
    loss_type="mse",
    patience=10,
    checkpoint_path=output_dir / Path("baseline0_ridge_alpha_%s.pt" % alpha_str),
)

ax = plot_training_history({'ridge': linear_history})

plt.savefig(output_dir / Path('baseline0_ridge_alpha_%s.png' % alpha_str), bbox_inches='tight')


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

linear_test = run_epoch(
    model=linear_model,
    dataloader=test_loader,
    gene_mean=gene_mean,
    gene_std=gene_std,
    device=DEVICE,
    optimizer=None,
)

print("Linear regression:")
print(linear_test)


##################################  PER-GENE METRICS  ###################################

linear_gene_metrics = evaluate_per_gene(
    linear_model,
    test_loader,
    gene_mean,
    gene_std,
    DEVICE,
)

linear_gene_metrics.to_csv(output_dir / Path('baseline0_ridge_alpha_%s.csv' % alpha_str), 
                           index=False)
