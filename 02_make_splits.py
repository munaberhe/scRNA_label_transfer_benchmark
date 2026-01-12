import numpy as np
import scanpy as sc

adata = sc.read_h5ad("data_pbmc3k_processed.h5ad")

labels = adata.obs["leiden"].to_numpy()
n_cells = adata.n_obs

rng = np.random.default_rng(seed=42)
indices = np.arange(n_cells)
rng.shuffle(indices)

train_size = n_cells // 2
train_idx = indices[:train_size]
test_idx = indices[train_size:]

adata.obs["split"] = "test"
adata.obs["split"].iloc[train_idx] = "train"

print(adata.obs["split"].value_counts())

adata.write("data_pbmc3k_splits.h5ad")
import numpy as np
import scanpy as sc

adata = sc.read_h5ad("data_pbmc3k_processed.h5ad")

labels = adata.obs["leiden"].to_numpy()
n_cells = adata.n_obs

rng = np.random.default_rng(seed=42)
indices = np.arange(n_cells)
rng.shuffle(indices)

train_size = n_cells // 2
train_idx = indices[:train_size]
test_idx = indices[train_size:]

adata.obs["split"] = "test"
adata.obs["split"].iloc[train_idx] = "train"

print(adata.obs["split"].value_counts())

adata.write("data_pbmc3k_splits.h5ad")

