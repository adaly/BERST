import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from lazy_datasets import VisiumArrayDataset
from lazy_datasets import get_dataset_dimensions, compute_gene_statistics
from baseline_models import SpatialGCNExpressionModel, SpatialGATExpressionModel
from training import train_expression_model, run_epoch
from training import evaluate_per_gene, evaluate_per_array, plot_training_history


################################  SPECIFY SPATIAL MODEL  ################################

import argparse
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-t', '--model-type', type=str, default='GCN', required=False,
                    help='Spatial model type ("GCN" or "GAT")')
parser.add_argument('-d', '--model-depth', type=int, default=1, required=False,
                    help='Spatial model depth (# blocks)')
parser.add_argument('-m', '--metadata', type=str, default="none", required=False,
                    help='Level of metadata to use in encoding -- "tissue", "all", or "none"')
parser.add_argument('-c', '--tissue-conditioning', action='store_true', required=False,
                    help='Whether to condition spatial layers on tissue_id via FiLM')
args = parser.parse_args()

MODEL_TYPE = args.model_type
MODEL_DEPTH = args.model_depth
METADATA = args.metadata
TISSUE_COND = args.tissue_conditioning
ACCUM_ITERS = 1

META_STR = METADATA + '_film1' if TISSUE_COND else METADATA

if METADATA not in ["none", "all", "tissue"]:
    raise ValueError('metadata flag must be set to "none", "tissue", or "all"')

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {DEVICE}")

output_dir = Path('/commons/groups/phatnani_lab/adaly/BERST/results/spatial_meta')

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

if METADATA != "none" or TISSUE_COND:
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

train_dataset = VisiumArrayDataset(root_dir, array_ids=train, return_metadata=True)
val_dataset = VisiumArrayDataset(root_dir, array_ids=val, return_metadata=True)
test_dataset = VisiumArrayDataset(root_dir, array_ids=test, return_metadata=True)

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

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)


##################################  INSTANTIATE MODEL  ##################################

# Initialize model
kwargs = {
    "metadata_mode": METADATA,
    "tissue_conditioning": TISSUE_COND,
}

if METADATA != "none" or TISSUE_COND:
    kwargs["categorical_cardinalities"] = categorical_cardinalities
    kwargs["tissue_index"] = tissue_index

if METADATA == "all":
    kwargs["n_continuous"] = n_continuous

if MODEL_TYPE == "GCN":
    spatial_model = SpatialGCNExpressionModel(
        input_dim=input_dim,
        n_genes=n_genes,
        n_spatial_blocks=MODEL_DEPTH,
        n_mlp_blocks=2,  # based on spotwise experiments
        hidden_dim=1024,
        latent_dim=256,
        dropout=0.2,
        **kwargs,
    )
elif MODEL_TYPE == "GAT":
    spatial_model = SpatialGATExpressionModel(
        input_dim=input_dim,
        n_genes=n_genes,
        n_spatial_blocks=MODEL_DEPTH,
        n_mlp_blocks=2,  # based on spotwise experiments
        hidden_dim=1024,
        latent_dim=256,
        heads=4,
        dropout=0.2,
        **kwargs,
    )
else:
    raise NotImplementedError

#####################################  TRAIN MODEL  #####################################

spatial_model, spatial_history = train_expression_model(
    model=spatial_model,
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
    checkpoint_path=output_dir/ Path("spatial_%s_b%d_%s.pt" % (MODEL_TYPE, MODEL_DEPTH, META_STR)),
    accum_iters=ACCUM_ITERS,
)

ax = plot_training_history({MODEL_TYPE: spatial_history})

plt.savefig(output_dir / Path("spatial_%s_b%d_%s.png" % (MODEL_TYPE, MODEL_DEPTH, META_STR)), 
            bbox_inches='tight')


#####################################  TEST MODEL  ######################################

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
)

spatial_test = run_epoch(
    model=spatial_model,
    dataloader=test_loader,
    gene_mean=gene_mean,
    gene_std=gene_std,
    device=DEVICE,
    optimizer=None,
)

print(MODEL_TYPE)
print(spatial_test)


###############################  PER-GENE/ARRAY METRICS  ################################

spatial_gene_metrics = evaluate_per_gene(
    spatial_model,
    test_loader,
    gene_mean,
    gene_std,
    DEVICE,
)

spatial_gene_metrics.to_csv(output_dir / Path("spatial_%s_b%d_%s_genemetrics.csv" % (MODEL_TYPE, MODEL_DEPTH, META_STR)), 
                            index=False)

spatial_array_metrics = evaluate_per_array(
    spatial_model,
    test_loader,
    gene_mean,
    gene_std,
    DEVICE,
)

spatial_array_metrics.to_csv(output_dir / Path("spatial_%s_b%d_%s_arraymetrics.csv" % (MODEL_TYPE, MODEL_DEPTH, META_STR)), 
                             index=False)


