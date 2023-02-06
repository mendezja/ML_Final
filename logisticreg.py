from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LogisticRegression
from distutils.version import LooseVersion
from eda import EDA

eda = EDA()
X_train, X_test, y_train, y_test = eda.train_test_split()
print(np.shape(X_train))

lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
pipeline = make_pipeline(MaxAbsScaler(), lr)
pipeline.fit(X_train, y_train) 

# Predict
y_pred = pipeline.predict(X_test)
# Probability of predicting true label
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]


# Print confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
# Print evaluation metrics
print("\n\nDecision Tree Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(
    y_test, y_pred, average="weighted"))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
print("Recall (weighted):", recall_score(y_test, y_pred, average="weighted"))
