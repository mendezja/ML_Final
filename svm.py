from eda import EDA
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, validation_curve
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, precision_score, recall_score, RocCurveDisplay
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier

# Get data from eda
eda = EDA()
X_train, X_test, y_train, y_test = eda.train_test_split()
# print(np.count_nonzero(y_train)/len(y_train))


# Create pipeline
# svcm = SVC(kernel="linear") # When not using ROC AUC
svcm = SVC(kernel="rbf",C=125, gamma='scale', decision_function_shape='ovo', probability=True)
pipeline = make_pipeline(MaxAbsScaler(), svcm)
 

# Fit data
pipeline.fit(X_train, y_train) 

# Predict
y_pred = pipeline.predict(X_test)
# Probability of predicting true label
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Print confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# Print evaluation metrics
print("\n\nSVM Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (weighted):", precision_score(
    y_test, y_pred, average="weighted"))
print("Recall (weighted):", recall_score(y_test, y_pred, average="weighted"))
print("ROC AUC (weighted):", roc_auc_score(
    y_test, y_pred_prob, multi_class="ovr", average="weighted"
))
print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))

RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("SVM")
plt.show()

# Rand search for optimal hyperparameters
# Other Potential hyperparameters: coef0: float, class_weight: dict or ‘balanced’, tol: float(default=1e-3), probability: bool, shrinking: bool
# param_grid = {"svc__C": [0.001, 0.01, 0.1, 1, 10, 100],
#               "svc__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
#               "svc__degree": [1, 2, 3, 4, 5],  # Only for 'poly' kernel
#               "svc__gamma": ["scale", "auto"],
#               "svc__decision_function_shape": ["ovo", "ovr"]}



# randSearch = RandomizedSearchCV(estimator=pipeline_gs, param_distributions=param_grid, verbose=3, n_jobs=-1, n_iter = 100, scoring='accuracy')
# randSearch.fit(X_train, y_train)
# randSearch_predictions = randSearch.predict(X_test)
# print(classification_report(y_test, randSearch_predictions))
# print(randSearch.best_params_)

# # use pipeline for random/grid search
# svcm_gs = SVC(kernel ='rbf', probability=True)
# pipeline_gs = make_pipeline(MaxAbsScaler(), svcm_gs)

# param_grid = {"svc__C": [50, 75, 100, 125],
#             #   "svc__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
#             #   "svc__degree": [1, 2, 3, 4, 5],  # Only for 'poly' kernel
#               "svc__gamma": ["scale", "auto"],
#               "svc__decision_function_shape": ["ovo", "ovr"]}

# grid = GridSearchCV(estimator=pipeline_gs, param_grid=param_grid, verbose=3, n_jobs=-1, scoring='accuracy')
# grid.fit(X_train, y_train)
# grid_predictions = grid.predict(X_test)

# print(classification_report(y_test, grid_predictions))
# print(grid.best_params_)
