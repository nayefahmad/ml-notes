# # Micro-, macro- and weighted-average scores for classification

# ## Overview

# ## References

# - [sklearn docs](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)  # noqa
# - [SE post](https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin)  # noqa


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

lr = LogisticRegressionCV(max_iter=1e8)
lr.fit(X_train, y_train)
preds = lr.predict(X_train)

scores = precision_recall_fscore_support(y_train, preds)
pd.DataFrame(scores).T.reset_index().rename(
    columns={0: "precision", 1: "recall", 2: "f1", 3: "support", "index": "class"}
)

print(classification_report(y_train, preds))

micro_precision = precision_score(y_train, preds, average="micro")
macro_precision = precision_score(y_train, preds, average="macro")

micro_recall = recall_score(y_train, preds, average="micro")
macro_recall = recall_score(y_train, preds, average="macro")
