# # Micro-, macro- and weighted-average scores for classification

# ## Overview

# ## References

# - [sklearn docs](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)  # noqa


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
import pandas as pd

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# ## Fit two models on training data and use CV scores to select a final model

# LR
lr = LogisticRegressionCV(max_iter=1e8)
lr.fit(X_train, y_train)

cross_val_score(lr, X_train, y_train, scoring="accuracy").mean()
pd.DataFrame(
    cross_validate(
        lr,
        X_train,
        y_train,
        return_train_score=True,
        scoring=(
            "precision_micro",
            "precision_macro",
            "precision_weighted",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
        ),
    )
)

# SVC
svc = LinearSVC()

# Selected model = _

# Evaluate on test set

# Discuss micro-, macro-, weighted-average results
