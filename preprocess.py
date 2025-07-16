import os
import pkgutil
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from io import StringIO
from scipy import sparse

# Preprocess raw count data for input to scBERT model:
# 1. Subset data to list of desired genes (those in gene2vec model, by default), imputing 0s for missing
# 2. Filter samples by min_depth and/or min_genes
# 3. Depth-normalize and log2(+1)-transform count data
def preprocess_scbert(adata, target_depth=1e4, counts_layer=None, min_genes=None, min_depth=None, 
	gene_symbols=None, target_genes=None):
	'''
	Parameters:
	----------
	adata: AnnData
		AnnData object containing raw count data
	target_depth: float
		number of counts to normalize each spot to
	counts_layer: str or None
		layer of adata containing raw counts, or "None" to default to adata.X
	min_genes: int or None
		filter spots with fewer than min_genes
	min_depth: int or None
		filter spots with fewer than min_counts (prior to depth normalization)
	gene_symbols: str or None
		column name in adata.var storing gene_symbols matching target_genes
	target_genes: path or None
		path to single-column CSV file containing ordered list of gene names to pull from adata,
		or "None" to default to the default list of gene2vec.
	'''
	if target_genes is None:
		ref_data = pkgutil.get_data('data', 'gene2vec_names.csv').decode('utf-8')
		ref_data = StringIO(ref_data)
	else:
		ref_data = target_genes
	ref_names = pd.read_csv(ref_data, header=None, index_col=0).index

	if counts_layer is None:
		X = adata.X
	else:
		X = adata.layers[counts_layer]
	ref = ref_names.tolist()

	counts = sparse.lil_matrix((X.shape[0],len(ref_names)),dtype=np.float32)

	# AnnData-based way of populating empty counts matrix:
	if gene_symbols is not None:
		var_old = adata.var.copy()
		adata.var[gene_symbols] = adata.var[gene_symbols].astype(str)
		adata.var = adata.var.set_index(gene_symbols)
		adata.var_names_make_unique()  # handles multiple ENSEMBL with same common name

	genes_shared = adata.var.index.intersection(ref)
	inds_shared = [ref.index(g) for g in genes_shared]
	counts[:, inds_shared] = adata[:, genes_shared].X

	new = ad.AnnData(X=counts.tocsr(), obs=adata.obs)
	new.var_names = ref

	for g in genes_shared:
		assert np.array_equal(
			np.array(adata[:,g].X.todense()).squeeze(), 
			np.array(new[:,g].X.todense()).squeeze())
	
	if gene_symbols is not None:
		adata.var = var_old  # undo modification of original AnnData

	if min_genes is not None or min_depth is not None:
		sc.pp.filter_cells(new, min_genes=min_genes, min_counts=min_depth)

	sc.pp.normalize_total(new, target_sum=target_depth)
	sc.pp.log1p(new, base=2)

	return new