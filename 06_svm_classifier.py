import scanpy as sc
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Make sure figures directory exists
Path("figures").mkdir(exist_ok=True)

# Load the dataset with train/test split
adata = sc.read_h5ad("data_pbmc3k_splits.h5ad")

# Features: PCA embedding
X = adata.obsm["X_pca"]
y = adata.obs["leiden"].astype("category")

# Train / test masks
train_mask = adata.obs["split"] == "train"
test_mask = adata.obs["split"] == "test"

X_train = X[train_mask.values, :]
X_test = X[test_mask.values, :]
y_train = y[train_mask].to_numpy()
y_test = y[test_mask].to_numpy()

# Define and train SVM classifier
svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Classification report
report = classification_report(y_test, y_pred)
print("Classification report (SVM):")
print(report)

with open("svm_classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=y.cat.categories)
cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in y.cat.categories],
    columns=[f"pred_{c}" for c in y.cat.categories],
)
print("Confusion matrix (SVM):")
print(cm_df)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("SVM confusion matrix")
plt.colorbar()
plt.xticks(range(len(y.cat.categories)), y.cat.categories, rotation=90)
plt.yticks(range(len(y.cat.categories)), y.cat.categories)
plt.tight_layout()
plt.savefig("figures/confusion_matrix_svm.png", dpi=200)
plt.close()


