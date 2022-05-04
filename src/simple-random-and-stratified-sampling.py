# # Exploring simple random sampling vs stratified sampling in classification

# ## Overview

# The goal here is to try to assess whether stratified sampling is helpful in cases
# where class imbalance is not extreme. In these cases, a simple random sample will
# probably be enough to ensure that all classes appear in the training data. In a large
# enough dataset, the class proportions in the training data should also be close to
# the true proportions in the population.

# **Approach**: simulate data for an imbalanced classification problem. Then compare
# simple random versus stratified sampling by fitting a logistic regression-based
# classifier in each case, and evaluating on a test set.

# I tried around 20 iterations of the simulation by manually changing the seed. In
# future, it would be better to try a larger number, and run them programmatically.

# *Conclusion*: There does not seem to be a consistent benefit to stratified sampling
# in this case.


# ## Libraries

from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support


# ## Define parameters for creating a classification problem


@dataclass(frozen=True)
class MakeClassificationParams:
    """Class for holding parameters required by sklearn.datasets.make_classification"""

    num_samples: int
    num_classes: int
    weights: list
    num_features: int
    n_informative: int
    n_repeated: int
    n_redundant: int


# ## Set parameters

classification_params = MakeClassificationParams(
    num_samples=10000,
    num_classes=2,
    weights=[0.95, 0.05],
    num_features=3,
    n_informative=2,
    n_repeated=0,
    n_redundant=0,
)

test_size_second_split = 0.90
seed = 202206

# ## Simulate data

X, y = make_classification(
    n_samples=classification_params.num_samples,
    n_classes=classification_params.num_classes,
    weights=classification_params.weights,
    n_features=classification_params.num_features,
    n_informative=classification_params.n_informative,
    n_redundant=classification_params.n_redundant,
    n_repeated=classification_params.n_repeated,
    random_state=seed,
)

fig, ax = plt.subplots()
ax.bar(["class 0", "class 1"], pd.Series(y).value_counts().to_list())
ax.set_title("Distribution of data across classes")
fig.show()

# Initial split into train/test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=seed
)

fig = plt.figure()
titles = [
    f"{dataset}: Distribution of data across classes"
    for dataset in ["Train set", "Test set"]
]
for i, target in enumerate([y_train, y_test]):
    ax = plt.subplot(2, 1, i + 1)
    ax.bar(["class 0", "class 1"], pd.Series(target).value_counts().to_list())
    ax.set_title(titles[i])
fig.tight_layout()
fig.show()

# Second split into train/test datasets

X_train_simple_random_sample, _, y_train_simple_random_sample, _ = train_test_split(
    X_train, y_train, test_size=test_size_second_split, shuffle=True, random_state=seed
)

X_train_stratified_sample, _, y_train_stratified_sample, _ = train_test_split(
    X_train,
    y_train,
    test_size=test_size_second_split,
    stratify=y_train,
    shuffle=True,
    random_state=seed,
)

# ## Modeling with logistic regression

# Model hyperparams
C = np.logspace(0, 4, 10)
penalty = ["l1", "l2"]
hyperparams = dict(C=C, penalty=penalty)

lr = LogisticRegression(solver="liblinear")

gridsearch = GridSearchCV(lr, hyperparams, cv=5, verbose=0)


# ## Analysis

# Case 1:
# grid-search CV on train_simple_random_sample, and evaluate on *_test

# plot
fig, ax = plt.subplots()
ax.bar(
    ["class 0", "class 1"],
    pd.Series(y_train_simple_random_sample).value_counts().to_list(),
)
ax.set_title("Simple random sample: Distribution of data across classes")
fig.show()

# CV
best_model_01 = gridsearch.fit(
    X_train_simple_random_sample, y_train_simple_random_sample
)

best_penalty = best_model_01.best_estimator_.get_params()["penalty"]
best_C = best_model_01.best_estimator_.get_params()["C"]
print(f"Coefs from simple random sampling: {best_model_01.best_estimator_.coef_}")

best_acc_score = best_model_01.best_estimator_.score(
    X_train_simple_random_sample, y_train_simple_random_sample
)
best_acc_score_test = best_model_01.best_estimator_.score(X_test, y_test)

y_pred_simple_random_train = best_model_01.best_estimator_.predict(
    X_train_simple_random_sample
)

best_results_train = precision_recall_fscore_support(
    y_train_simple_random_sample, y_pred_simple_random_train, average="binary"
)

y_pred_simple_random_sample_test = best_model_01.best_estimator_.predict(X_test)

best_results_test = precision_recall_fscore_support(
    y_test, y_pred_simple_random_sample_test, average="binary"
)

print("")
# print(f'Simple random sampling: Best accuracy score on train data: {best_acc_score}')
# print(f'Simple random sampling: Best accuracy score on test data: {best_acc_score_test}')  # noqa
# print(f'Simple random sampling: Best accuracy precision, recall, f1 on train: {best_results_train}')  # noqa
# print(f'Simple random sampling: Best accuracy precision, recall, f1 on test: {best_results_test}')  # noqa


# Case 2:
# grid-search CV on train_simple_random_sample, and evaluate on _test

# plot
fig, ax = plt.subplots()
ax.bar(
    ["class 0", "class 1"],
    pd.Series(y_train_stratified_sample).value_counts().to_list(),
)
ax.set_title("Stratified sample: Distribution of data across classes")
fig.show()

# CV
best_model_02 = gridsearch.fit(X_train_stratified_sample, y_train_stratified_sample)

best_penalty_02 = best_model_02.best_estimator_.get_params()["penalty"]
best_C_02 = best_model_02.best_estimator_.get_params()["C"]
print(f"Coefs from stratified sampling:    {best_model_02.best_estimator_.coef_}")

best_acc_score_02 = best_model_02.best_estimator_.score(
    X_train_stratified_sample, y_train_stratified_sample
)
best_acc_score_test_02 = best_model_02.best_estimator_.score(X_test, y_test)

y_pred_stratified_train = best_model_02.best_estimator_.predict(
    X_train_stratified_sample
)

best_results_02_train = precision_recall_fscore_support(
    y_train_stratified_sample, y_pred_stratified_train, average="binary"
)

y_pred_stratified_test = best_model_02.best_estimator_.predict(X_test)

best_results_02_test = precision_recall_fscore_support(
    y_test, y_pred_stratified_test, average="binary"
)


# # Summary
print("")

print(f"X.shape:       {X.shape}")
print(f"X_train.shape: {X_train.shape}")
print(f"X_train_simple_random_sample.shape: {X_train_simple_random_sample.shape}")
print(f"X_train_stratified_sample.shape:    {X_train_stratified_sample.shape}")

print(f"Simple random sampling: Best accuracy score on train data: {best_acc_score}")
print(f"Simple random sampling: Best accuracy score on test data:{best_acc_score_test}")

print(f"Stratified sampling: Best accuracy score on train data:{best_acc_score_02}")
print(f"Stratified sampling: Best accuracy score on test data:{best_acc_score_test_02}")

print(
    f"Simple random sampling: Best precision, recall, f1 on train data: {best_results_train}"  # noqa
)
print(
    f"Simple random sampling: Best precision, recall, f1 on test data:  {best_results_test}"  # noqa
)

print(
    f"Stratified sampling: Best precision, recall, f1 on train data:    {best_results_02_train}"  # noqa
)
print(
    f"Stratified sampling: Best precision, recall, f1 on test data:     {best_results_02_test}"  # noqa
)
