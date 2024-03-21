"""
# Nested CV for hyperparam tuning and generalization error estimation

## Overview

When using CV to both find the best hyperparams, and to estimate generalization error,
we use nested CV

## References:
# 1. [sklearn course](https://inria.github.io/scikit-learn-mooc/python_scripts/cross_validation_nested.html)  # noqa
# 2. [Christoph Molnar's blog](https://mindfulmodeler.substack.com/p/how-to-get-from-evaluation-to-final)  # noqa
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

data, target = load_breast_cancer(return_X_y=True)

model = SVC()

param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1]}

search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
search.fit(data, target)

print(f"best params: {search.best_params_}")
print(f"mean CV score using best params: {search.best_score_:.3f}")


"""
Now that we have identified best params on this data, how do we estimate generalization
error of the model that uses these params?

- Option 1: Use the `search.best_score` value above. This is equivalent to the
  following: Instantiate a new SVC with these params. Use CV on the same data that was
  used for hyperparam tuning.
    - Problem: Score will be too optimistic. The same data is being used to select the
      best hyperparams and to evaluate generalization error. We used knowledge from the
      test sets (i.e. test scores) to select the hyper-parameter of the model itself.

- Option 2: Use a “nested” cross-validation. Use an inner CV loop to select best
  hyperparams (similar to what was done above), and an outer CV loop to evaluate
  testing/generalization error. Basically, think of the "outer" CV loop as standard CV,
  but with the added feature that for any particular iteration of the loop, the
  hyperparameters selected and used for predicting on the test fold were selected using
  CV on the train set, and trying out several different hyperparameter values.
"""

print("done")
