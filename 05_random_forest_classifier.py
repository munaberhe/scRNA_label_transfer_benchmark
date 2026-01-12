import scanpy as sc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Make sure figures folder exists
Path("figures").mkdir(exist_ok=True)

# 1. Load the split data
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

# 2. Fit RandomForest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# 3. Predict on test set
y_pred = rf.predict(X_test)

# 4. Classification report
report = classification_report(y_test, y_pred)
print("Classification report (RandomForest):")
print(report)

# Optionally save the report to a text file for reference
with open("rf_classification_report.txt", "w") as f:
    f.write(report)

# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=y.cat.categories)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in y.cat.categories],
    columns=[f"pred_{c}" for c in y.cat.categories],
)

print("Confusion matrix (RandomForest):")
print(cm_df)

# 6. Save confusion matrix as an image
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("RandomForest confusion matrix")
plt.colorbar()
plt.xticks(range(len(y.cat.categories)), y.cat.categories, rotation=90)
plt.yticks(range(len(y.cat.categories)), y.cat.categories)
plt.tight_layout()
plt.savefig("figures/confusion_matrix_randomforest.png", dpi=200)
plt.close()

