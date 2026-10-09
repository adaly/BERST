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

output_dir = Path('/commons/groups/phatnani_lab/adaly/BERST/results/spotwise_mlp')

n_blocks = int(sys.argv[1])

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

class MLPExpressionModel(nn.Module):
    """
    Configurable MLP for predicting gene expression
    from pretrained image features.

    Architecture:
        Input (input_dim)
            |
        [Linear -> LayerNorm -> GELU -> Dropout]
            |             repeated n_blocks times
            |
        Linear -> latent_dim
            |
        Expression head -> n_genes

    Parameters
    ----------
    input_dim : int
        Number of input image features.

    n_genes : int
        Number of target genes.

    n_blocks : int
        Number of hidden blocks controlling model complexity.

    hidden_dim : int
        Width of each hidden block.

    latent_dim : int
        Dimension of the final image representation.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim,
        n_genes,
        n_blocks=2,
        hidden_dim=1024,
        latent_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        if n_blocks < 0:
            raise ValueError("n_blocks must be nonnegative")

        layers = [
            nn.LayerNorm(input_dim)
        ]

        current_dim = input_dim

        # -----------------------------------------
        # Hidden blocks
        # -----------------------------------------

        for _ in range(n_blocks):

            layers.extend([
                nn.Linear(
                    current_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

            current_dim = hidden_dim

        # -----------------------------------------
        # Final latent projection
        # -----------------------------------------

        layers.append(
            nn.Linear(
                current_dim,
                latent_dim,
            )
        )

        layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)

        # -----------------------------------------
        # Gene expression prediction
        # -----------------------------------------

        self.expression_head = nn.Linear(
            latent_dim,
            n_genes,
        )

    def forward(self, x):

        z = self.encoder(x)

        return self.expression_head(z)

# Initialize model
mlp_model = MLPExpressionModel(
    input_dim=input_dim,
    n_genes=n_genes,
    n_blocks=n_blocks,
    hidden_dim=1024,
    latent_dim=256,
    dropout=0.2,
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
    checkpoint_path=output_dir/ Path("baseline1_mlp_b%d.pt" % n_blocks),
)

ax = plot_training_history({'MLP': mlp_history})

plt.savefig(output_dir / Path('baseline1_mlp_b%d.png' % n_blocks), bbox_inches='tight')


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

print("MLP:")
print(mlp_test)


##################################  PER-GENE METRICS  ###################################

mlp_gene_metrics = evaluate_per_gene(
    mlp_model,
    test_loader,
    gene_mean,
    gene_std,
    DEVICE,
)

mlp_gene_metrics.to_csv(output_dir / Path('baseline1_mlp_b%d.csv' % n_blocks), index=False)
