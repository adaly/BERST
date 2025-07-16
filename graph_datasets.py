import os
import torch
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import issparse
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder

from squidpy.gr import spatial_neighbors
from torch_geometric.data import Data, InMemoryDataset


#########################################################
#        Visium AnnData to torch_geometric Data			#
#########################################################

# Create a single (graph) Data object out of a Visium AnnData
# - If the Visium AnnData contains multiple samples/batches, will treat as a single graph w/unconnected subgraphs
def visium_anndata_to_graphdata(adata, spatial_key='spatial', annot_col='annotation', annot_encoding=None, 
                                x_layer=None, x_obsm=None, **kwargs):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm containing node coordinates
    annot_col: str
        column in adata.obs containing spot/node-level annotations
    annot_encoding: list of str or None
        enumeration of all possible annotation classes to be used in integer coding.
        when None, defaults to ordered list of all annotations seen in adata.obs
    x_layer: str
        entry in adata.layers to use for node features (default: adata.X)
    x_obsm: str
        entry in adata.obsm to use for node features -- supercedes x_layer
    kwargs: additional arguments for compute_adjacency_radius

    Returns:
    -------
    gdat: torch_geometric.data.Data
        graph Data representation of adata
    '''
    # Access nodel-level features
    if x_layer is not None:
        X = adata.layers[x_layer]
    elif x_obsm is not None:
        X = adata.obsm[x_obsm]
    else:
        X = adata.X
    if issparse(X):
        X = X.todense()
    X = torch.tensor(X, dtype=torch.float)
    
    # Encode node-level labels
    if annot_encoding is not None:
        y = [list(annot_encoding).index(a) for a in adata.obs[annot_col].values]
    else:
        le = LabelEncoder().fit(adata.obs[annot_col])
        annot_encoding = le.classes_
        y = le.transform(adata.obs[annot_col])
    y = torch.tensor(y, dtype=torch.long)

    # Compute adjacencies and edge weights
    A, w = compute_adjacency_radius(adata, spatial_key, **kwargs)
    if A is None or w is None:
        return None
    A = torch.tensor(A, dtype=torch.long)
    w = torch.tensor(w, dtype=torch.float).T

    # Extract spot coordinates
    c = torch.tensor(adata.obsm[spatial_key], dtype=torch.float)
    
    gdat = Data(x=X, edge_index=A, edge_attr=w, y=y, pos=c)
    return gdat

# Create an iterable of (graph) Data objects, one for each sample (batch) in a Visium AnnData
# - If the Visium AnnData contains multiple samples/batches, treats each as a separate graph (to be passed to DataLoader)
def visium_anndata_to_graphdataset(adata, spatial_key='spatial', batch_col='array', annot_col='annotation', annot_encoding=None, 
                                   x_layer=None, x_obsm=None, **kwargs):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm containing node coordinates
    batch_col: str
        column in adata.obs denoting individual samples/batches, each of which will comprise a separate graph
    annot_col: str
        column in adata.obs containing spot/node-level annotations
    annot_encoding: list of str or None
        enumeration of all possible annotation classes to be used in integer coding.
        when None, defaults to ordered list of all annotations seen in adata.obs
    x_layer: str
        entry in adata.layers to use for node features (default: adata.X)
    x_obsm: str
        entry in adata.obsm to use for node features -- supercedes x_layer
    kwargs: additional arguments for compute_adjacency_radius

    Returns:
    -------
    gdat_list: list of torch_geometric.data.Data
        list of Data representing each individual sample/batch in the AnnData
    '''
    # Create a unified encoding for annotations (if not provided)
    if annot_encoding is None:
        le = LabelEncoder().fit(adata.obs[annot_col])
        annot_encoding = le.classes_

    # Create separate graph data object for each independent batch
    gdat_list, batch_names = [], []
    for b in adata.obs[batch_col].unique():
        adata_sub = adata[adata.obs[batch_col]==b].copy()
        if len(adata_sub) > 0:
            gdat = visium_anndata_to_graphdata(adata_sub, spatial_key, annot_col, annot_encoding, x_layer, x_obsm, **kwargs)
            if gdat is None or not gdat.validate():
                print('Skipping batch %s (%d nodes) -- invalid graph' % (b, len(adata_sub)))
            else:
                gdat_list.append(gdat)
                batch_names.append(b)
            
    return gdat_list, batch_names

'''
Because graphs can have variables numbers of nodes/edges, we cannot use basic PyTorch Datasets to do minibatching.

Instead, Geometric provides two abstract base classes:
1. InMemoryDataset -- for datasets small enough to fit in CPU memory
2. Dataset -- for datasets that should be partially read into CPU memory

Using a subset of these classes, a DataLoader can create mini-batches by:
- Block-diagonal stacking the adjacency matrices (easy in COO sparse format), and
- Concatenating node/target features in the node dimension
https://pytorch-geometric.readthedocs.io/en/2.6.1/advanced/batching.html
'''
class GraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.', transform=None, pre_transform=None)
        self.data, self.slices = self.collate(data_list)

    def get(self, idx):
        return super().get(idx)

#########################################################
#        Squidy-based adjacency calculations			#
#########################################################

# Copies node coordinates to a desired key in adata.obsm (required for Squidpy's spatial_neighbors)
def add_spatial(adata, spatial_key, x_col, y_col, pseudo_hex=False):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm in which to store node coordinates
    x_col, y_col: str, str
        columns in adata.obs containing x and y coordinates (respectively) of nodes; only needed if spatial_key not set
    pseudo_hex: bool
        whether coordinates follow Visium's pseudo-hex scheme (default False: cartesian coordinates)
        if True, coordinates will be transformed to Cartesian for storage in adata.obsm
    '''
    # Store node locations in adata.obsm
    for c in [x_col, y_col]:
        if c not in adata.obs:
            raise ValueError('No entry for %s in obs metadata' % c)
    true_coords = adata.obs[[x_col, y_col]].values.copy().astype(float)
    if pseudo_hex:
        true_coords[:,0] = true_coords[:,0] * 0.5          # halve horizontal dimension
        true_coords[:,1] = true_coords[:,1] * np.sqrt(3)/2 # scale vertical dimension to achieve unit adjacencies
    adata.obsm[spatial_key] = true_coords

    return adata

# Compute adjacencies for an AnnData object potentially spanning multiple samples (batches)
def compute_adjacency_radius(adata, spatial_key='spatial', x_col=None, y_col=None, batch_col=None, 
                             radius=1.01, unit_dist_um=100, pseudo_hex=False):
    '''
    Parameters:
    ----------
    adata: AnnData
    spatial_key: str
        key in adata.obsm in which to find/store node coordinates
    x_col, y_col: str, str
        columns in adata.obs containing x and y coordinates (respectively) of nodes (if spatial_key not set)
    batch_col: str
        column in adata.obs denoting independent samples (e.g., Visium arrays)
    radius: float
        radius in coordinate system within which nodes are considered neighbors 
        (default 1.01 to account for rounding errors in pseudo-visium conversion)
    unit_dist_um: float
        physical distance (in um) corresponding to unit distance in coordinate system
    pseudo_hex: bool
        whether coordinates follow Visium's pseudo-hex scheme (default False: cartesian coordinates)
        if True, coordinates will be transformed to Cartesian for storage in adata.obsm

    Returns:
    -------
    A: [2, num_edges] ndarray
        computed edges in COO format
    w: [num_edges,] ndarray
        physical distances (um) corresponding to each edge
    '''
    if spatial_key not in adata.obsm:
        adata = add_spatial(adata, spatial_key, x_col, y_col, pseudo_hex)
    else:
        assert adata.obsm[spatial_key].shape == (len(adata), 2)
    C, D = spatial_neighbors(adata, spatial_key=spatial_key, library_key=batch_col, coord_type='generic', radius=radius, copy=True)
    max_k = C.sum(axis=0).max()
    if max_k == 0:
        return None, None
    
    A = C.nonzero()
    w = D[A] * unit_dist_um
    return np.array(A), w
