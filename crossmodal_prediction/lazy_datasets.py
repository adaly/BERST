import os
import json
import bisect
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree

from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

#########################################################################################
##                              PREPROCESSING FUNCTIONS                                ##
#########################################################################################

'''
Assume:
- adata contains all arrays.
- adata.obs["array_id"] identifies arrays.
- adata.obs_names are spot IDs.
- Each image-feature CSV has spots as rows and the image features as columns.
'''
def preprocess_array(
    adata,
    array_id,
    image_feature_file,
    target_genes,
    output_dir,
    array_key="array_id",
    target_sum=1e4,
    clip_max=10.0,
):
    """
    Preprocess one Visium array and save it as memory-mappable .npy files.

    Expression processing:
        raw counts
        -> normalize by full-transcriptome library size
        -> subset target genes
        -> log1p
        -> clip

    Assumes:
        - adata.obs_names contain spot IDs
        - image feature CSV has spot IDs as its index
    """

    output_dir = Path(output_dir) / str(array_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Select expression data for this array
    # --------------------------------------------------

    mask = np.asarray(adata.obs[array_key] == array_id)
    ad = adata[mask]

    # --------------------------------------------------
    # 2. Read image features
    # --------------------------------------------------

    image_df = pd.read_csv(image_feature_file, index_col=0)

    # Find spots present in both modalities
    common_spots = ad.obs_names.intersection(image_df.index)

    if len(common_spots) == 0:
        raise ValueError(f"No matching spots found for {array_id}")

    # Keep exactly the same spot order
    ad = ad[common_spots]
    image_df = image_df.loc[common_spots]

    # --------------------------------------------------
    # 3. Full-transcriptome library size
    # --------------------------------------------------

    X = ad.X

    if sp.issparse(X):
        library_size = np.asarray(X.sum(axis=1)).ravel()
    else:
        library_size = np.asarray(X.sum(axis=1)).ravel()

    # Avoid divide-by-zero
    library_size = np.maximum(library_size, 1)

    # --------------------------------------------------
    # 4. Extract target genes
    # --------------------------------------------------

    missing = set(target_genes) - set(ad.var_names)

    if missing:
        raise ValueError(
            f"{len(missing)} target genes missing from AnnData: "
            f"{sorted(missing)[:10]}"
        )

    Y = ad[:, target_genes].X

    if sp.issparse(Y):
        Y = Y.toarray()

    Y = np.asarray(Y, dtype=np.float32)

    # --------------------------------------------------
    # 4b. Compute pseudobulk expression
    # --------------------------------------------------
    
    bulk_counts = Y.sum(axis=0)
    
    bulk_library_size = library_size.sum()
    
    bulk_expression = (
        target_sum
        * bulk_counts
        / bulk_library_size
    )

    # No clipping on bulk data -- aggregation should be enough to combat noise
    bulk_expression = np.log1p(
        bulk_expression
    ).astype(np.float32)

    # --------------------------------------------------
    # 5. Normalize / log / clip
    # --------------------------------------------------

    Y *= (target_sum / library_size[:, None])

    Y = np.log1p(Y)

    if clip_max is not None:
        Y = np.clip(Y, 0, clip_max)

    # --------------------------------------------------
    # 6. Image features
    # --------------------------------------------------

    X_image = image_df.to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # 7. Coordinates, if available
    # --------------------------------------------------

    if "spatial" in ad.obsm:
        coords = np.asarray(ad.obsm["spatial"], dtype=np.float32)
        np.save(output_dir / "coords.npy", coords)

        # Find all pairs of spots within radius 1.1.
        # query_pairs returns each undirected pair once.
        tree = cKDTree(coords)
    
        pairs = tree.query_pairs(
            r=1.1,
            output_type="ndarray",
        )
    
        # Convert [E, 2] undirected pairs into PyG's
        # directed [2, 2E] edge_index representation.
        if len(pairs) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
        else:
            edge_index = np.concatenate(
                [
                    pairs,
                    pairs[:, ::-1],
                ],
                axis=0,
            ).T.astype(np.int64)
    
        np.save(
            output_dir / "edge_index.npy",
            edge_index,
        )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    np.save(output_dir / "image_features.npy", X_image)
    np.save(output_dir / "expression.npy", Y)
    np.save(
        output_dir / "library_size.npy",
        library_size.astype(np.float32),
    )

    # Save both scaled/transformed bulk expression and raw bulk counts
    # (in order to trial different losses/constraints)
    np.save(
        output_dir / "bulk_expression.npy",
        bulk_expression,
    )
    np.save(
        output_dir / "bulk_counts.npy",
        bulk_counts.astype(np.float32),
    )

    # strings don't need to be mmap'd
    pd.Series(
        common_spots,
        name="spot_id",
    ).to_csv(output_dir / "spot_ids.csv", index=False)

    metadata = {
        "array_id": str(array_id),
        "n_spots": len(common_spots),
        "n_image_features": X_image.shape[1],
        "n_genes": len(target_genes),
        "target_sum": target_sum,
        "clip_max": clip_max,
        "target_genes": list(target_genes),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"{array_id}: "
        f"{len(common_spots):,} spots, "
        f"{X_image.shape[1]} image features, "
        f"{Y.shape[1]} genes"
    )

'''
One-time creation of:
    global index -> array -> spot index
to enable lazy data loading
'''
def build_dataset_index(root_dir, array_ids=None):

    root_dir = Path(root_dir)

    if array_ids is None:
        array_dirs = sorted(
            p for p in root_dir.iterdir()
            if p.is_dir()
        )
    else:
        array_dirs = [
            root_dir / str(x)
            for x in array_ids
        ]

    records = []

    for array_dir in array_dirs:

        with open(array_dir / "metadata.json") as f:
            metadata = json.load(f)

        n = metadata["n_spots"]

        records.append({
            "array_id": metadata["array_id"],
            "path": str(array_dir),
            "n_spots": n,
        })

    return records


#########################################################################################
##                                DATASET/SAMPLER CLASSES                              ##
#########################################################################################

'''
Base dataset for spot-level inference
'''
class VisiumSpotDataset(Dataset):

    def __init__(
        self,
        root_dir,
        array_ids=None,
        return_coords=False,
        return_library_size=False,
        return_metadata=False,
        tissue_index=1,
    ):

        self.root_dir = Path(root_dir)

        self.arrays = build_dataset_index(
            root_dir,
            array_ids=array_ids,
        )

        self.return_coords = return_coords
        self.return_library_size = return_library_size
        self.return_metadata = return_metadata

        # cumulative END positions
        #
        # e.g. array sizes [100, 200, 50]
        # -> [100, 300, 350]
        self.cumulative_sizes = np.cumsum([
            x["n_spots"]
            for x in self.arrays
        ])

        # Cache mmap handles, NOT the actual data.
        self._cache = {}

        # Index in categorical metadata containing tissue type
        self.tissue_index = tissue_index

    def __len__(self):
        return int(self.cumulative_sizes[-1])

    def _locate(self, idx):

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        array_idx = bisect.bisect_right(
            self.cumulative_sizes,
            idx,
        )

        previous_end = (
            0
            if array_idx == 0
            else self.cumulative_sizes[array_idx - 1]
        )

        local_idx = idx - previous_end

        return array_idx, int(local_idx)

    def _load_array(self, array_idx):

        if array_idx in self._cache:
            return self._cache[array_idx]

        info = self.arrays[array_idx]
        path = Path(info["path"])

        data = {
            "image": np.load(
                path / "image_features.npy",
                mmap_mode="r",
            ),
            "expression": np.load(
                path / "expression.npy",
                mmap_mode="r",
            ),
        }

        if self.return_coords:
            data["coords"] = np.load(
                path / "coords.npy",
                mmap_mode="r",
            )

        if self.return_library_size:
            data["library_size"] = np.load(
                path / "library_size.npy",
                mmap_mode="r",
            )

        if self.return_metadata:
            data['categorical_metadata'] = np.load(
                path / "categorical_metadata.npy",
                mmap_mode='r',
            )
            data['continuous_metadata'] = np.load(
                path / "continuous_metadata.npy",
                mmap_mode='r',
            )
            data['continuous_metadata_mask'] = np.load(
                path / "continuous_metadata_mask.npy",
                mmap_mode='r',
            )

        self._cache[array_idx] = data

        return data

    def __getitem__(self, idx):

        array_idx, local_idx = self._locate(idx)

        data = self._load_array(array_idx)

        item = {
            "image": torch.from_numpy(
                np.array(data["image"][local_idx])
            ).float(),

            "expression": torch.from_numpy(
                np.array(data["expression"][local_idx])
            ).float(),

            "array_idx": array_idx,
            "spot_idx": local_idx,
        }

        if self.return_coords:
            item["coords"] = torch.from_numpy(
                np.array(data["coords"][local_idx])
            ).float()

        if self.return_library_size:
            item["library_size"] = torch.tensor(
                data["library_size"][local_idx],
                dtype=torch.float32,
            )

        if self.return_metadata:
            item["categorical_metadata"] = torch.tensor(
                data["categorical_metadata"],
                dtype=torch.long,
            )
            item["continuous_metadata"] = torch.tensor(
                data["continuous_metadata"],
                dtype=torch.float32,
            )
            item["continuous_metadata_mask"] = torch.tensor(
                data["continuous_metadata_mask"],
                dtype=torch.float32,
            )
            # Special field indicating tissue type
            item['tissue_id'] = item['categorical_metadata'][self.tissue_index]

        return item

class VisiumArrayDataset(Dataset):
    """
    Array-level dataset for spatial expression prediction.

    One Dataset item = one Visium array.

    Expected directory structure
    ----------------------------
    root_dir/
        array_001/
            image_features.npy   # [N, input_dim]
            expression.npy       # [N, G], unstandardized log1p expression
            coords.npy           # [N, 2]
            library_size.npy     # optional
        array_002/
            ...

    Parameters
    ----------
    root_dir : str or Path
        Root processed-data directory.

    array_ids : sequence
        Arrays included in this Dataset.

    gene_mean, gene_std : array-like, optional
        Training-set gene normalization parameters.
        If supplied, expression is standardized before being returned.

    k : int
        Number of nearest spatial neighbors.

    graph_loop : bool
        Whether knn_graph should explicitly include self-loops.

    cache_arrays : int
        Number of fully constructed graphs to cache in memory.
        Keep small for large arrays.
    """

    def __init__(
        self,
        root_dir,
        array_ids,
        k=6,
        graph_loop=False,
        cache_arrays=2,
        return_metadata=False,
        tissue_index=1,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.array_ids = list(array_ids)

        self.arrays = []

        for array_id in self.array_ids:
            path = self.root_dir / str(array_id)
        
            with open(path / "metadata.json", "r") as f:
                metadata = json.load(f)
        
            self.arrays.append({
                "array_id": str(array_id),
                "path": path,
                "n_spots": metadata["n_spots"],
            })

        self.k = k
        self.graph_loop = graph_loop

        self.cache_arrays = cache_arrays
        self._cache = OrderedDict()

        self.return_metadata = return_metadata

        # Index in categorical metadata containing tissue type
        self.tissue_index = tissue_index

    def __len__(self):
        return len(self.array_ids)

    def _load_array(self, idx):

        array_id = self.array_ids[idx]

        # ------------------------------------------
        # Cache lookup
        # ------------------------------------------

        if array_id in self._cache:
            self._cache.move_to_end(array_id)
            return self._cache[array_id]

        path = self.root_dir / array_id

        # ------------------------------------------
        # Load arrays
        # ------------------------------------------

        image = torch.from_numpy(
            np.array(
                np.load(
                    path / "image_features.npy",
                    mmap_mode="r",
                ),
                dtype=np.float32,
                copy=True,
            )
        )

        expression = torch.from_numpy(
            np.array(
                np.load(
                    path / "expression.npy",
                    mmap_mode="r",
                ),
                dtype=np.float32,
                copy=True,
            )
        )

        coords = torch.from_numpy(
            np.array(
                np.load(
                    path / "coords.npy",
                    mmap_mode="r",
                ),
                dtype=np.float32,
                copy=True,
            )
        )

        bulk_expression = torch.from_numpy(
            np.array(
                np.load(
                    path / "bulk_expression.npy",
                    mmap_mode="r",
                ),
                dtype=np.float32,
                copy=True,
            )
        )

        library_size = torch.from_numpy(
            np.array(
                np.load(
                    path / "library_size.npy",
                    mmap_mode="r",
                ),
                dtype=np.float32,
                copy=True,
            )
        )

        if self.return_metadata:
            categorical_metadata = torch.from_numpy(
                np.array(
                    np.load(
                        path / "categorical_metadata.npy",
                        mmap_mode="r",
                    ),
                    dtype=np.int64,
                    copy=True,
                )
            )

            continuous_metadata = torch.from_numpy(
                np.array(
                    np.load(
                        path / "continuous_metadata.npy",
                        mmap_mode="r",
                    ),
                    dtype=np.float32,
                    copy=True,
                )
            )

            continuous_metadata_mask = torch.from_numpy(
                np.array(
                    np.load(
                        path / "continuous_metadata_mask.npy",
                        mmap_mode="r",
                    ),
                    dtype=np.float32,
                    copy=True,
                )
            )

        # ------------------------------------------
        # Construct spatial graph
        # ------------------------------------------

        edge_file = path / "edge_index.npy"

        if edge_file.exists():
            edge_index = torch.from_numpy(
                np.array(
                    np.load(edge_file, mmap_mode="r"),
                    dtype=np.int64,
                    copy=True,
                )
            )
        else:
            edge_index = radius_graph(
                coords,
                r=1.1,
                loop=self.graph_loop,
                max_num_neighbors=self.k,
            )
            
            edge_index = to_undirected(
                edge_index,
                num_nodes=coords.shape[0],
            )

        edge_index = torch.unique(
            edge_index,
            dim=1,
        )

        # ------------------------------------------
        # PyG Data object
        # ------------------------------------------

        data = Data(
            x=image,
            y=expression,
            pos=coords,
            edge_index=edge_index,
            bulk_expression=bulk_expression,
            library_size=library_size,
        )

        data.array_id = array_id

        # Optional library sizes
        library_file = path / "library_size.npy"

        if library_file.exists():

            data.library_size = torch.from_numpy(
                np.array(
                    np.load(
                        library_file,
                        mmap_mode="r",
                    ),
                    dtype=np.float32,
                    copy=True,
                )
            )

        # Optional array metadata
        if self.return_metadata:
            
            # Unsqueeze to allow proper handling of batch dimensions
            data.categorical_metadata = categorical_metadata.unsqueeze(0)
            data.continuous_metadata = continuous_metadata.unsqueeze(0)
            data.continuous_metadata_mask = continuous_metadata_mask.unsqueeze(0)

            # Special field containing tissue type
            data.tissue_id = categorical_metadata[self.tissue_index].unsqueeze(0)

        # ------------------------------------------
        # LRU cache
        # ------------------------------------------

        if self.cache_arrays > 0:

            self._cache[array_id] = data

            while len(self._cache) > self.cache_arrays:
                self._cache.popitem(last=False)

        return data

    def __getitem__(self, idx):
        return self._load_array(idx)


'''
Custom BatchSampler that accomplishes shuffling in an array-aware manner, 
taking advantage of the caching in VisiumSpotDataset:
1. Randomly shuffle arrays at the beginning of each epoch
2. Randomly shuffle spots within an array
3. Generate batches containing only spots from one array
4. Process all batches from one array before moving to the next
5. (Optionally) shuffle order of batches within an array
'''
class ArrayBatchSampler(Sampler):
    """
    Generate batches of spot indices grouped by Visium array.

    Each epoch:
        1. Shuffle the order of arrays.
        2. Shuffle spots within each array.
        3. Generate batches from each array.
        4. Move to the next array.

    All indices in a batch belong to the same array.

    Parameters
    ----------
    dataset : VisiumSpotDataset
        Dataset exposing cumulative_sizes.

    batch_size : int
        Number of spots per batch.

    shuffle_arrays : bool
        Whether to shuffle array order each epoch.

    shuffle_spots : bool
        Whether to shuffle spot order within each array.

    shuffle_batches : bool
        Whether to shuffle the order of batches within an array.

    drop_last : bool
        Whether to drop incomplete batches from each array.

    seed : int
        Random seed used for reproducible shuffling.
    """

    def __init__(
        self,
        dataset,
        batch_size=512,
        shuffle_arrays=True,
        shuffle_spots=True,
        shuffle_batches=False,
        drop_last=False,
        seed=42,
    ):

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size

        self.shuffle_arrays = shuffle_arrays
        self.shuffle_spots = shuffle_spots
        self.shuffle_batches = shuffle_batches

        self.drop_last = drop_last
        self.seed = seed

        self.epoch = 0

        # Starting global index of each array
        #
        # Example:
        # cumulative_sizes = [100, 300, 350]
        # array_starts     = [0, 100, 300]

        self.array_starts = np.concatenate([
            [0],
            dataset.cumulative_sizes[:-1],
        ])

        self.array_sizes = np.asarray([
            array_info["n_spots"]
            for array_info in dataset.arrays
        ])

    def set_epoch(self, epoch):
        """
        Set the epoch for reproducible, epoch-dependent shuffling.
        """
        self.epoch = epoch

    def __iter__(self):

        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        n_arrays = len(self.array_sizes)

        # ---------------------------------------------
        # 1. Determine array order
        # ---------------------------------------------

        array_order = np.arange(n_arrays)

        if self.shuffle_arrays:
            rng.shuffle(array_order)

        # ---------------------------------------------
        # 2. Iterate through arrays
        # ---------------------------------------------

        for array_idx in array_order:

            array_start = int(
                self.array_starts[array_idx]
            )

            n_spots = int(
                self.array_sizes[array_idx]
            )

            # -----------------------------------------
            # 3. Generate local spot indices
            # -----------------------------------------

            spot_indices = np.arange(n_spots)

            if self.shuffle_spots:
                rng.shuffle(spot_indices)

            # Convert local indices to global indices
            spot_indices += array_start

            # -----------------------------------------
            # 4. Split into batches
            # -----------------------------------------

            batches = []

            for start in range(
                0,
                n_spots,
                self.batch_size,
            ):

                batch = spot_indices[
                    start:start + self.batch_size
                ]

                if (
                    self.drop_last
                    and len(batch) < self.batch_size
                ):
                    continue

                batches.append(batch.tolist())

            # -----------------------------------------
            # 5. Optionally shuffle batch order
            # -----------------------------------------

            if self.shuffle_batches:
                rng.shuffle(batches)

            # -----------------------------------------
            # 6. Yield batches from this array
            # -----------------------------------------

            for batch in batches:
                yield batch

    def __len__(self):

        if self.drop_last:

            return int(
                np.sum(
                    self.array_sizes // self.batch_size
                )
            )

        return int(
            np.sum(
                (
                    self.array_sizes
                    + self.batch_size
                    - 1
                ) // self.batch_size
            )
        )


#########################################################################################
##                                HELPER/UTILITY FUNCTIONS                             ##
#########################################################################################

def get_dataset_dimensions(dataset):
    """
    Infer image-feature dimension and number of target genes.

    Supports:
        - spotwise dataset returning dicts with
          ["image"] and ["expression"]
        - array-wise/PyG dataset returning Data objects
          with .x and .y
    """

    sample = dataset[0]

    is_geometric = (
        hasattr(sample, "x")
        and hasattr(sample, "y")
        and hasattr(sample, "edge_index")
    )

    if is_geometric:
        # x: [N_spots, N_features]
        # y: [N_spots, N_genes]
        input_dim = sample.x.shape[-1]
        n_genes = sample.y.shape[-1]

    else:
        # image: [N_features]
        # expression: [N_genes]
        input_dim = sample["image"].numel()
        n_genes = sample["expression"].numel()

    return input_dim, n_genes
    
def compute_gene_statistics(
    dataset,
    min_std=1e-6,
):
    """
    Compute global, spot-weighted per-gene mean/std.

    Uses per-array expression files directly when the Dataset
    exposes .arrays; otherwise falls back to iterating through
    Dataset items.

    Works with both spotwise and array-wise datasets.
    """

    total_n = 0
    total_sum = None
    total_sum_sq = None

    def accumulate(X):
        nonlocal total_n
        nonlocal total_sum
        nonlocal total_sum_sq

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim == 1:
            X = X[None, :]

        if total_sum is None:

            total_sum = np.zeros(
                X.shape[1],
                dtype=np.float64,
            )

            total_sum_sq = np.zeros(
                X.shape[1],
                dtype=np.float64,
            )

        elif X.shape[1] != len(total_sum):

            raise ValueError(
                "Inconsistent number of genes "
                "across dataset."
            )

        total_n += X.shape[0]

        total_sum += X.sum(
            axis=0
        )

        total_sum_sq += np.square(
            X
        ).sum(axis=0)

    # --------------------------------------------------
    # Fast path: load each array only once
    # --------------------------------------------------

    if hasattr(dataset, "arrays"):

        for array_info in dataset.arrays:

            path = Path(
                array_info["path"]
            )

            expression = np.load(
                path / "expression.npy",
                mmap_mode="r",
            )

            accumulate(expression)

    # --------------------------------------------------
    # Generic fallback
    # --------------------------------------------------

    else:

        for sample in dataset:

            if hasattr(sample, "y"):
                expression = sample.y
            else:
                expression = sample["expression"]

            if torch.is_tensor(expression):
                expression = (
                    expression
                    .detach()
                    .cpu()
                    .numpy()
                )

            accumulate(expression)

    # --------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------

    if total_n == 0:
        raise ValueError(
            "Dataset contains no spots."
        )

    gene_mean = (
        total_sum / total_n
    )

    gene_var = (
        total_sum_sq / total_n
        - gene_mean**2
    )

    gene_var = np.maximum(
        gene_var,
        0.0,
    )

    gene_std = np.sqrt(
        gene_var
    )

    gene_std = np.maximum(
        gene_std,
        min_std,
    )

    return (
        torch.tensor(
            gene_mean,
            dtype=torch.float32,
        ),
        torch.tensor(
            gene_std,
            dtype=torch.float32,
        ),
    )