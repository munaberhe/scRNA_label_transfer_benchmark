import scanpy as sc

# 1. Load dataset
adata = sc.datasets.pbmc3k()  # downloads & caches automatically

print(adata)
print(adata.obs.head())

# 2. Do a quick standard preprocessing & clustering so we get labels
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.5)  # this gives clusters

print(adata.obs['leiden'].value_counts())

# 3. Save processed data for later steps
adata.write("data_pbmc3k_processed.h5ad")

