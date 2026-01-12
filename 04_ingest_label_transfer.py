import scanpy as sc
from sklearn.metrics import classification_report

# 1. Load the split data
adata = sc.read_h5ad("data_pbmc3k_splits.h5ad")

# 2. Split into reference (train) and query (test)
ref = adata[adata.obs["split"] == "train"].copy()
query = adata[adata.obs["split"] == "test"].copy()

# 3. Save the true labels from the query set BEFORE ingest overwrites them
query.obs["leiden_true"] = query.obs["leiden"].astype("category")

# 4. Ensure ref has PCA, neighbors, and UMAP (required for ingest)
if "X_pca" not in ref.obsm:
    sc.tl.pca(ref, svd_solver="arpack")

sc.pp.neighbors(ref, n_neighbors=10, n_pcs=40, use_rep="X_pca")
sc.tl.umap(ref)

# 5. Run ingest: this will map ref.obs["leiden"] onto query.obs["leiden"]
sc.tl.ingest(query, ref, obs="leiden")

# 6. Evaluate performance
y_true = query.obs["leiden_true"].astype("category")
y_pred = query.obs["leiden"].astype("category")  # now holds predicted labels

print("Classification report (Scanpy ingest):")
print(classification_report(y_true, y_pred))

# 7. UMAP visualisation: true vs predicted labels on the query set
sc.pl.umap(
    query,
    color=["leiden_true", "leiden"],
    wspace=0.4,
    show=False,                     # <- add this
    save="_query_true_vs_pred.png",
)

