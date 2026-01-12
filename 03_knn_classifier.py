import scanpy as sc
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

adata = sc.read_h5ad("data_pbmc3k_splits.h5ad")

# Features: PCA coordinates
X = adata.obsm["X_pca"]
y = adata.obs["leiden"].astype("category")

train_mask = adata.obs["split"] == "train"
test_mask  = adata.obs["split"] == "test"

X_train = X[train_mask.values, :]
X_test  = X[test_mask.values, :]
y_train = y[train_mask].to_numpy()
y_test  = y[test_mask].to_numpy()

# Fit kNN
clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=y.cat.categories)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in y.cat.categories],
    columns=[f"pred_{c}" for c in y.cat.categories],
)

print("Confusion matrix:")
print(cm_df)

# Save confusion matrix as image
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("kNN confusion matrix")
plt.colorbar()
plt.xticks(range(len(y.cat.categories)), y.cat.categories, rotation=90)
plt.yticks(range(len(y.cat.categories)), y.cat.categories)
plt.tight_layout()
plt.savefig("figures_knn_confusion_matrix.png", dpi=200)
plt.close()

