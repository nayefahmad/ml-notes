# # Micro-, macro- and weighted-average scores for classification

# ## Overview

# ## References

# - [sklearn docs](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)  # noqa
# - [SE post](https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin)  # noqa


from typing import List

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

print(classification_report(y_train, preds, digits=4))

micro_precision = precision_score(y_train, preds, average="micro")
macro_precision = precision_score(y_train, preds, average="macro")
weighted_precision = precision_score(y_train, preds, average="weighted")
print(micro_precision, macro_precision, weighted_precision)

micro_recall = recall_score(y_train, preds, average="micro")
macro_recall = recall_score(y_train, preds, average="macro")
weighted_recall = recall_score(y_train, preds, average="weighted")
print(micro_recall, macro_recall, weighted_recall)


def count_true_positives_and_false_negatives(
    *, _class: int, true_values: List, predicted_values: List
) -> int:
    class_values_filter = [True if x == _class else False for x in true_values]
    class_values = [
        item for item, include in zip(true_values, class_values_filter) if include
    ]
    class_predicted = [
        item for item, include in zip(predicted_values, class_values_filter) if include
    ]
    true_positives = [
        True if pred == actual else False
        for pred, actual in zip(class_predicted, class_values)
    ]

    # false_negatives = []  # todo: finish this

    count = sum(true_positives)
    return count


def count_false_positives(*, _class: int, true_values: List, predicted_values: List):
    class_predicted_filter = [True if x == _class else False for x in predicted_values]
    class_predicted = [
        item
        for item, include in zip(predicted_values, class_predicted_filter)
        if include
    ]
    class_actual = [
        item for item, include in zip(true_values, class_predicted_filter) if include
    ]

    false_positives = [
        True if pred != actual else False
        for pred, actual in zip(class_predicted, class_actual)
    ]
    count = sum(false_positives)
    return count


for target in range(3):
    tp = count_true_positives_and_false_negatives(
        _class=target, true_values=y_train, predicted_values=preds
    )
    fp = count_false_positives(
        _class=target, true_values=y_train, predicted_values=preds
    )

    print(f"TP = {tp}")
    print(f"FP = {fp}")
