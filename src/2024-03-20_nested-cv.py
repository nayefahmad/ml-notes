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
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVC

data, target = load_breast_cancer(return_X_y=True)

model_to_tune = SVC()

param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1]}

search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, cv=5, n_jobs=-1)
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

inner_cv = KFold(n_splits=5, shuffle=True, random_state=2024)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=2024)

# set up inner CV using GridSearchCV object
model = GridSearchCV(
    estimator=model_to_tune, param_grid=param_grid, cv=inner_cv, n_jobs=-1
)

"""
Setting cv=outer_cv ensures that a test set is set aside, while the train set
that is passed to `model` (aka a GridSearchCV) is then further split using the
cv attribute of that object. This ensures that for each of the three outer_cv
iterations, the inner CV is used to select best hyperparams, and then the model with
those hyperparams is used to predict on the test set from outer_cv, which it has never
seen before.
"""
test_set_scores = cross_val_score(model, data, target, cv=outer_cv, n_jobs=-1)

print("done")
