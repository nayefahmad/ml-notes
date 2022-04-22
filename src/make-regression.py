# # Exploring scikit-learn's make_regression() function

import pandas as pd
import numpy as np
from sklearn.datasets._samples_generator import make_low_rank_matrix, util_shuffle
from src.utils import check_random_state


def make_regression(
    n_samples=100,
    n_features=100,
    *,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=True,
    coef=False,
    random_state=None,
):
    """Generate a random regression problem.
    The input set can either be well conditioned (by default) or have a low
    rank-fat tail singular profile. See :func:`make_low_rank_matrix` for
    more details.
    The output is generated by applying a (potentially biased) random linear
    regression model with `n_informative` nonzero regressors to the previously
    generated input and some gaussian centered noise with some adjustable
    scale.
    Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=100
        The number of features.
    n_informative : int, default=10
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.
    n_targets : int, default=1
        The number of regression targets, i.e., the dimension of the y output
        vector associated with a sample. By default, the output is a scalar.
    bias : float, default=0.0
        The bias term in the underlying linear model.
    effective_rank : int, default=None
        if not None:
            The approximate number of singular vectors required to explain most
            of the input data by linear combinations. Using this kind of
            singular spectrum in the input allows the generator to reproduce
            the correlations often observed in practice.
        if None:
            The input set is well conditioned, centered and gaussian with
            unit variance.
    tail_strength : float, default=0.5
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None. When a float, it should be
        between 0 and 1.
    noise : float, default=0.0
        The standard deviation of the gaussian noise applied to the output.
    shuffle : bool, default=True
        Shuffle the samples and the features.
    coef : bool, default=False
        If True, the coefficients of the underlying linear model are returned.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The output values.
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        The coefficient of the underlying linear model. It is returned only if
        coef is True.
    """
    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    if effective_rank is None:
        # Randomly generate a well conditioned input set
        X = generator.randn(n_samples, n_features)

    else:
        # Randomly generate a low rank, fat tail input set
        X = make_low_rank_matrix(
            n_samples=n_samples,
            n_features=n_features,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            random_state=generator,
        )

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = np.zeros((n_features, n_targets))
    ground_truth[:n_informative, :] = 100 * generator.rand(n_informative, n_targets)

    y = np.dot(X, ground_truth) + bias

    # Add noise
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    y = np.squeeze(y)

    if coef:
        return X, y, np.squeeze(ground_truth)

    else:
        return X, y


if __name__ == "__main__":
    X, y = make_regression(n_samples=10, n_features=3)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    df = pd.concat([df_X, df_y], axis=1)
    df.columns = [f"x_{i}" for i in range(3)] + ["y"]
    print(df)
