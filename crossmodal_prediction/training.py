import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


#########################################################################################
##                                 TRAINING FUNCTIONS                                  ##
#########################################################################################

def run_epoch(
    model,
    dataloader,
    gene_mean,
    gene_std,
    device,
    optimizer=None,
    ridge_alpha=0.0,
    loss_type="mse",
    bulk_weight=0.0,
    target_sum=1e4,
    accum_iters=1,
):
    """
    Run one training or evaluation epoch.

    If optimizer is None, performs evaluation.

    Expression values returned by the Dataset are
    globally standardized before calculating the loss.

    Returns
    -------
    dict containing:
        loss: mean spot-level regression loss
        rmse: aggregate standardized-expression RMSE
        mae: aggregate standardized-expression MAE
    """

    is_training = optimizer is not None

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    model.train(is_training)

    gene_mean = gene_mean.to(device)
    gene_std = gene_std.to(device)

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    accum_iters = int(accum_iters)
    if accum_iters <= 0:
        raise ValueError("accum_iters must be a positive integer")

    for batch_idx, batch in enumerate(dataloader):
        is_geometric = (
            hasattr(batch, "edge_index")
            and hasattr(batch, "x")
            and hasattr(batch, "y")
        )

        if is_geometric:
            # PyTorch Geometric batch.
            #
            # .to(device) moves x, y, edge_index, pos, etc.
            # together.
            batch = batch.to(
                device,
                non_blocking=True,
            )

            x = batch.x
            y = batch.y

            metadata_kwargs = {}

            if hasattr(batch, "categorical_metadata"):
                metadata_kwargs["categorical_metadata"] = (
                    batch.categorical_metadata
                )
                metadata_kwargs["continuous_metadata"] = (
                    batch.continuous_metadata
                )
                metadata_kwargs["continuous_metadata_mask"] = (
                    batch.continuous_metadata_mask
                )
        
            if hasattr(batch, "tissue_id"):
                metadata_kwargs["tissue_id"] = batch.tissue_id

        else:
            # Original spot-wise dictionary batch.
            x = batch["image"].to(
                device,
                non_blocking=True,
            )

            y = batch["expression"].to(
                device,
                non_blocking=True,
            )

            metadata_kwargs = {}

            if "categorical_metadata" in batch:
                metadata_kwargs["categorical_metadata"] = (
                    batch["categorical_metadata"].to(
                        device, non_blocking=True
                    )
                )
                metadata_kwargs["continuous_metadata"] = (
                    batch["continuous_metadata"].to(
                        device, non_blocking=True
                    )
                )
                metadata_kwargs["continuous_metadata_mask"] = (
                    batch["continuous_metadata_mask"].to(
                        device, non_blocking=True
                    )
                )
        
            if "tissue_id" in batch:
                metadata_kwargs["tissue_id"] = (
                    batch["tissue_id"].to(
                        device, non_blocking=True
                    )
                )

        # -----------------------------------------
        # Global gene-wise standardization
        # -----------------------------------------

        y = (y - gene_mean) / gene_std

        # -----------------------------------------
        # Forward pass
        # -----------------------------------------

        with torch.set_grad_enabled(is_training):

            if is_geometric:
                pred = model(
                    x, 
                    batch.edge_index,
                    batch=getattr(batch, "batch", None),
                    **metadata_kwargs)
            else:
                pred = model(x, **metadata_kwargs)

            if loss_type == "mse":

                regression_loss = F.mse_loss(
                    pred,
                    y,
                )

            elif loss_type == "huber":

                regression_loss = F.huber_loss(
                    pred,
                    y,
                    delta=1.0,
                )

            else:

                raise ValueError(
                    f"Unknown loss type: {loss_type}"
                )

            loss = regression_loss

            # -------------------------------------
            # Optional bulk reconstruction loss
            # -------------------------------------
            
            if bulk_weight > 0:
                
                if not is_geometric:
                    raise ValueError('Bulk loss requires array-level batches')

                pred_log = pred * gene_std + gene_mean
                pred_norm = torch.expm1(pred_log).clamp_min(0)  # prevents negative counts

                library_size = batch.library_size
                
                pred_bulk_norm = (
                    pred_norm * library_size[:, None]
                ).sum(dim=0) / library_size.sum()

                pred_bulk = torch.log1p(pred_bulk_norm)

                bulk_loss = F.mse_loss(
                    pred_bulk,
                    batch.bulk_expression,
                )

                loss += bulk_weight * bulk_loss

            # -------------------------------------
            # Optional ridge penalty
            # -------------------------------------

            if is_training and ridge_alpha > 0:

                if not hasattr(model, "linear"):
                    raise ValueError(
                        "Ridge penalty requires a "
                        "model with a linear layer."
                    )

                l2_penalty = (
                    model.linear.weight.square().sum()
                )

                loss = (
                    loss
                    + ridge_alpha * l2_penalty
                )

            # -------------------------------------
            # Backpropagation
            # -------------------------------------

            if is_training:

                loss = loss / accum_iters
                loss.backward()

                do_step = (
                    (batch_idx + 1) % accum_iters == 0 
                    or (batch_idx + 1) == len(dataloader)
                )

                if do_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------
        # Accumulate regression metrics
        # -----------------------------------------

        with torch.no_grad():

            error = pred - y

            total_squared_error += (
                error.square().sum().item()
            )

            total_absolute_error += (
                error.abs().sum().item()
            )

            total_elements += y.numel()

    if total_elements == 0:
        raise ValueError("Dataloader produced no observations.")

    mse = total_squared_error / total_elements

    return {
        "loss": mse,
        "rmse": np.sqrt(mse),
        "mae": total_absolute_error / total_elements,
    }


def train_expression_model(
    model,
    train_loader,
    val_loader,
    gene_mean,
    gene_std,
    device,
    n_epochs=100,
    learning_rate=1e-3,
    weight_decay=0.0,
    ridge_alpha=0.0,
    loss_type="mse",
    patience=10,
    checkpoint_path=None,
    accum_iters=1,
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(n_epochs):

        # -----------------------------------------
        # Update array-aware sampler
        # -----------------------------------------

        sampler = getattr(
            train_loader,
            "batch_sampler",
            None,
        )

        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        # -----------------------------------------
        # Training
        # -----------------------------------------

        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            gene_mean=gene_mean,
            gene_std=gene_std,
            device=device,
            optimizer=optimizer,
            ridge_alpha=ridge_alpha,
            loss_type=loss_type,
            accum_iters=accum_iters,
        )

        # -----------------------------------------
        # Validation
        # -----------------------------------------

        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            gene_mean=gene_mean,
            gene_std=gene_std,
            device=device,
            optimizer=None,
            ridge_alpha=0.0,
            loss_type="mse",
        )

        # -----------------------------------------
        # Record metrics
        # -----------------------------------------

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train MSE: {train_metrics['loss']:.5f} | "
            f"Val MSE: {val_metrics['loss']:.5f} | "
            f"Val RMSE: {val_metrics['rmse']:.5f}"
        )

        # -----------------------------------------
        # Early stopping
        # -----------------------------------------

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            if checkpoint_path is not None:

                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": best_epoch,
                        "val_loss": best_val_loss,
                        "gene_mean": gene_mean.cpu(),
                        "gene_std": gene_std.cpu(),
                    },
                    checkpoint_path,
                )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping at epoch {epoch + 1}"
            )

            break

    # ---------------------------------------------
    # Restore best validation model
    # ---------------------------------------------

    if best_state is not None:
        model.load_state_dict(best_state)

    print(
        f"Best epoch: {best_epoch} | "
        f"Validation MSE: {best_val_loss:.5f}"
    )

    return model, history


#########################################################################################
##                                EVALUATION FUNCTIONS                                 ##
#########################################################################################

@torch.no_grad()
def evaluate_per_gene(
    model,
    dataloader,
    gene_mean,
    gene_std,
    device,
):
    """
    Calculate per-gene metrics in standardized
    log-expression space.

    Returns a pandas DataFrame with one row per gene.
    """

    import pandas as pd

    model.eval()

    gene_mean = gene_mean.to(device)
    gene_std = gene_std.to(device)

    n_genes = gene_mean.numel()

    # Accumulate in float64 for numerical stability
    sum_y = torch.zeros(n_genes, dtype=torch.float64)
    sum_pred = torch.zeros_like(sum_y)

    sum_y2 = torch.zeros_like(sum_y)
    sum_pred2 = torch.zeros_like(sum_y)

    sum_cross = torch.zeros_like(sum_y)

    sum_squared_error = torch.zeros_like(sum_y)
    sum_absolute_error = torch.zeros_like(sum_y)

    n_spots = 0

    for batch in dataloader:

        is_geometric = (
            hasattr(batch, "edge_index")
            and hasattr(batch, "x")
            and hasattr(batch, "y")
        )

        metadata_kwargs = {}

        if is_geometric:
            # PyTorch Geometric batch.
            #
            # .to(device) moves x, y, edge_index, pos, etc.
            # together.
            batch = batch.to(
                device,
                non_blocking=True,
            )

            x = batch.x
            y = batch.y

            if hasattr(batch, "categorical_metadata"):
                metadata_kwargs["categorical_metadata"] = (
                    batch.categorical_metadata
                )
                metadata_kwargs["continuous_metadata"] = (
                    batch.continuous_metadata
                )
                metadata_kwargs["continuous_metadata_mask"] = (
                    batch.continuous_metadata_mask
                )
        
            if hasattr(batch, "tissue_id"):
                metadata_kwargs["tissue_id"] = batch.tissue_id

        else:
            # Original spot-wise dictionary batch.
            x = batch["image"].to(
                device,
                non_blocking=True,
            )

            y = batch["expression"].to(
                device,
                non_blocking=True,
            )

            if "categorical_metadata" in batch:
                metadata_kwargs["categorical_metadata"] = (
                    batch["categorical_metadata"].to(
                        device, non_blocking=True
                    )
                )
                metadata_kwargs["continuous_metadata"] = (
                    batch["continuous_metadata"].to(
                        device, non_blocking=True
                    )
                )
                metadata_kwargs["continuous_metadata_mask"] = (
                    batch["continuous_metadata_mask"].to(
                        device, non_blocking=True
                    )
                )
        
            if "tissue_id" in batch:
                metadata_kwargs["tissue_id"] = (
                    batch["tissue_id"].to(
                        device, non_blocking=True
                    )
                )

        y = (y - gene_mean) / gene_std

        if is_geometric:
            pred = model(
                x, 
                batch.edge_index, 
                batch=getattr(batch, "batch", None),
                **metadata_kwargs,
            )
        else:
            pred = model(x, **metadata_kwargs)

        y = y.double().cpu()
        pred = pred.double().cpu()

        error = pred - y

        n_spots += y.shape[0]

        sum_y += y.sum(dim=0)
        sum_pred += pred.sum(dim=0)

        sum_y2 += y.square().sum(dim=0)
        sum_pred2 += pred.square().sum(dim=0)

        sum_cross += (y * pred).sum(dim=0)

        sum_squared_error += error.square().sum(dim=0)
        sum_absolute_error += error.abs().sum(dim=0)

    # ---------------------------------------------
    # Pearson correlation
    # ---------------------------------------------

    covariance = (
        sum_cross
        - sum_y * sum_pred / n_spots
    )

    variance_y = (
        sum_y2
        - sum_y.square() / n_spots
    )

    variance_pred = (
        sum_pred2
        - sum_pred.square() / n_spots
    )

    denominator = torch.sqrt(
        variance_y.clamp_min(0)
        * variance_pred.clamp_min(0)
    )

    pearson = covariance / denominator.clamp_min(1e-12)

    # Undefined when either variable is constant
    pearson[denominator <= 1e-12] = float("nan")

    # ---------------------------------------------
    # Regression metrics
    # ---------------------------------------------

    rmse = torch.sqrt(
        sum_squared_error / n_spots
    )

    mae = sum_absolute_error / n_spots

    return pd.DataFrame({
        "gene_idx": np.arange(n_genes),
        "pearson": pearson.numpy(),
        "rmse": rmse.numpy(),
        "mae": mae.numpy(),
    })


@torch.no_grad()
def evaluate_per_array(
    model,
    dataloader,
    gene_mean,
    gene_std,
    device,
    target_genes=None,
):
    """
    Evaluate an array-wise spatial model.

    For each array, computes:

    Spot-level:
        - MSE across all spot x gene values
        - MAE across all spot x gene values

    Within-array spatial performance:
        - Pearson correlation across spots for each gene
        - mean/median Pearson correlation across genes

    Array-level expression:
        - Pearson correlation across genes between predicted
          and observed pseudobulk expression

    Assumes DataLoader batch_size=1.
    """

    model.eval()

    gene_mean = gene_mean.to(device)
    gene_std = gene_std.to(device)

    results = []

    for batch in dataloader:

        batch = batch.to(
            device,
            non_blocking=True,
        )

        x = batch.x
        metadata_kwargs = {}

        if hasattr(batch, "categorical_metadata"):
            metadata_kwargs["categorical_metadata"] = (
                batch.categorical_metadata
            )
            metadata_kwargs["continuous_metadata"] = (
                batch.continuous_metadata
            )
            metadata_kwargs["continuous_metadata_mask"] = (
                batch.continuous_metadata_mask
            )
    
        if hasattr(batch, "tissue_id"):
            metadata_kwargs["tissue_id"] = batch.tissue_id

        # ------------------------------------------
        # Predict standardized expression
        # ------------------------------------------

        pred = model(
            x, 
            batch.edge_index, 
            batch=getattr(batch, "batch", None),
            **metadata_kwargs,
        )

        # batch.y is assumed to contain unstandardized
        # log1p-normalized expression.
        observed = batch.y

        # Convert predictions back to same scale.
        pred_log = (
            pred * gene_std
            + gene_mean
        )

        # ------------------------------------------
        # Basic spot-level metrics
        # ------------------------------------------

        error = pred_log - observed

        mse = (
            error.square()
            .mean()
            .item()
        )

        mae = (
            error.abs()
            .mean()
            .item()
        )

        # Move to NumPy for correlation calculations.
        pred_np = (
            pred_log
            .detach()
            .cpu()
            .numpy()
        )

        obs_np = (
            observed
            .detach()
            .cpu()
            .numpy()
        )

        # ------------------------------------------
        # Per-gene correlations within this array
        # ------------------------------------------

        gene_correlations = []

        for g in range(obs_np.shape[1]):

            obs_g = obs_np[:, g]
            pred_g = pred_np[:, g]

            # Pearson is undefined if either vector
            # has zero variance.
            if (
                np.std(obs_g) == 0
                or np.std(pred_g) == 0
            ):
                r = np.nan

            else:
                r = np.corrcoef(
                    obs_g,
                    pred_g,
                )[0, 1]

            gene_correlations.append(r)

        gene_correlations = np.asarray(
            gene_correlations
        )

        # ------------------------------------------
        # Predicted pseudobulk
        # ------------------------------------------

        pred_norm = torch.expm1(
            pred_log
        ).clamp_min(0)

        library_size = batch.library_size

        pred_bulk_norm = (
            pred_norm
            * library_size[:, None]
        ).sum(dim=0) / library_size.sum()

        pred_bulk = torch.log1p(
            pred_bulk_norm
        )

        # ------------------------------------------
        # Observed pseudobulk
        # ------------------------------------------

        if hasattr(batch, "bulk_expression"):

            observed_bulk = (
                batch.bulk_expression
                .reshape(-1)
            )

            pred_bulk_flat = (
                pred_bulk.reshape(-1)
            )

            bulk_r = np.corrcoef(
                pred_bulk_flat.cpu().numpy(),
                observed_bulk.cpu().numpy(),
            )[0, 1]

            bulk_mse = (
                (pred_bulk_flat - observed_bulk)
                .square()
                .mean()
                .item()
            )

        else:
            bulk_r = np.nan
            bulk_mse = np.nan

        # ------------------------------------------
        # Array ID
        # ------------------------------------------

        array_id = batch.array_id

        if isinstance(array_id, (list, tuple)):
            array_id = array_id[0]

        # ------------------------------------------
        # Store summary
        # ------------------------------------------

        row = {
            "array_id": array_id,
            "n_spots": obs_np.shape[0],

            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mae,

            "mean_gene_pearson": np.nanmean(
                gene_correlations
            ),

            "median_gene_pearson": np.nanmedian(
                gene_correlations
            ),

            "bulk_pearson": bulk_r,
            "bulk_mse": bulk_mse,
        }

        # Optionally retain individual gene correlations.
        if target_genes is not None:

            for gene, r in zip(
                target_genes,
                gene_correlations,
            ):
                row[f"pearson_{gene}"] = r

        results.append(row)

    return pd.DataFrame(results)


def plot_training_history(
    histories,
    metric="loss",
):
    """
    Parameters
    ----------
    histories : dict
        Mapping model names to training histories.

    metric : str
        One of "loss", "rmse", or "mae".
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, history in histories.items():

        df = pd.DataFrame(history)

        ax.plot(
            df["epoch"],
            df[f"train_{metric}"],
            label=f"{model_name} (train)",
            linestyle="--",
        )

        ax.plot(
            df["epoch"],
            df[f"val_{metric}"],
            label=f"{model_name} (validation)",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.upper())

    ax.legend()
    ax.grid(alpha=0.2)
    
    return ax
