from eda import EDA
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, precision_score, recall_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Get data from eda
eda = EDA()
X_train, X_test, y_train, y_test = eda.train_test_split()

# Create model
clf = DecisionTreeClassifier(criterion='gini', max_depth=18, max_features='sqrt',
                             min_samples_split=2, min_weight_fraction_leaf=0.1, splitter='random')

# # Fit data
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
# Predict probability
y_pred_prob = clf.predict_proba(X_test)

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


# print("ROC AUC (weighted):", roc_auc_score(
#     y_test, y_pred_prob, multi_class="ovr", average="weighted"
# ))

# # Plotting the decision tree
# plt.figure(figsize=(30, 10), facecolor='k')
# # Plot tree
# a = plot_tree(clf,
#               feature_names=eda.columns,
#               class_names=True,
#               rounded=True,
#               filled=True,
#               fontsize=14)
# plt.show()


# Grid search for optimal hyperparameters
# parameters = {"criterion": ["gini", "entropy", "log_loss"],
#               "splitter": ["best", "random"],
#               "min_samples_split": [2, 3, 4, 6, 8, 10],
#               "min_weight_fraction_leaf": [0.0, 0.0025, 0.025, 0.05, 0.1],
#               "max_features": ["sqrt", "log2", None]}
# grid = GridSearchCV(DecisionTreeClassifier(random_state=1),
#                     parameters, verbose=1, n_jobs=-1, scoring='accuracy')
# grid.fit(X_train, y_train)
# grid_predictions = grid.predict(X_test)

# print classification report
# print(classification_report(y_test, grid_predictions))
# print(grid.best_params_)


# Validation curve for max-depth
param_range = np.arange(2, 50, 1, dtype=int)
train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(criterion='gini', random_state=1), X_train, y_train,
    param_name="max_depth", param_range=param_range,
    cv=5, scoring='accuracy')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.title("Validation Curve")
plt.xlabel("Max-depth")
plt.ylabel("Accuracy")
lw = 2
plt.plot(
    param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.show()


# # Random forest and grid search
rf = RandomForestClassifier(criterion='gini', max_features='sqrt',
                            min_samples_split=8, min_weight_fraction_leaf=0.0, n_estimators=100)
rf.fit(X_train, y_train)
# Predict
y_pred_rfc = rf.predict(X_test)
# Predict probability
y_pred_prob_rfc = rf.predict_proba(X_test)

# # Print confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rfc)
plt.show()
# # Print evaluation metrics
print("\n\nRandom Forest Performance")
print("Accuracy:", accuracy_score(y_test, y_pred_rfc))
print("Precision (weighted):", precision_score(
    y_test, y_pred_rfc, average="weighted"))
print("Recall (weighted):", recall_score(
    y_test, y_pred_rfc, average="weighted"))
print("ROC AUC (weighted):", roc_auc_score(
    y_test, y_pred_prob_rfc, multi_class="ovr", average="weighted"
))
print("F1 (weighted):", f1_score(y_test, y_pred_rfc, average="weighted"))

# Grid search for Random Forest optimal hyperparameters
# parameters = {"n_estimators": [10, 50, 100, 500, 1000],
#               "criterion": ["gini", "entropy", "log_loss"],
#               'min_samples_split': [2, 4, 6, 8, 10],
#               'min_weight_fraction_leaf': [0.0, 0.0025, 0.025, 0.05, 0.075, 0.1, 0.15],
#               'max_features': ['sqrt', 'log2', None]
#               }

# Specify same parameters in RandomForest as DecisionTree for comparison
# rf = RandomForestClassifier(random_state=1)
# grid = GridSearchCV(rf, parameters, verbose=1, n_jobs=-1, scoring='accuracy')
# grid.fit(X_train, y_train.flatten())
# grid_predictions = grid.predict(X_test)

#print classification report
#print(grid.best_params_)
