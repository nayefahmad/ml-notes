"""
# Exploring TF-IDF

References:
    - V. Boykis, "Embeddings" doc.
    - [PCA notes](https://www.datacamp.com/tutorial/principal-component-analysis-in-python)  # noqa

"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Represent two tails by their history of destinations
tails = [
    "YUL, YYZ, YUL, YYZ, JFK",
    "JFK, LAX, JFK, LAX, YYZ, YVR, JFK",
    "DAC, CGP, DAC, BKK, DAC, BKK",
    "JFK, BKK, SIN, BKK, SIN, BKK, SIN, BKK, SIN, BKK, SIN, BKK",
]
tail_ids = [f"tail_0{x}" if x < 10 else f"tail_{x}" for x in range(1, 5)]


# Bag of words
vectorizer_01 = CountVectorizer()
vector_01 = vectorizer_01.fit_transform(tails)
assert vector_01.shape == (4, 9)

df_01 = pd.DataFrame(
    vector_01.toarray(), index=tail_ids, columns=vectorizer_01.get_feature_names()
).T


# TF-IDF
vectorizer_02 = TfidfVectorizer()
vector_02 = vectorizer_02.fit_transform(tails)
assert vector_02.shape == (4, 9)

dict(zip(vectorizer_02.get_feature_names(), vector_02.toarray()))

df_02 = pd.DataFrame(
    vector_02.toarray(), index=tail_ids, columns=vectorizer_02.get_feature_names()
).T

print(df_01)
print(df_02)


# ## PCA to reduce each tail to a smaller number of dimensions

seed = 2024

# Using bag-of-words features:
pca = Pipeline(
    [
        ("scaling", StandardScaler()),
        ("pca", PCA(random_state=seed)),
    ]
)

pca_01 = pca.fit(df_01.T)
pca_01.named_steps["pca"].explained_variance_

"""
PCA can extract at most min(n_samples, n_features) components. So here we can only
extract 4 components.

PC matrix has shape: (n_samples, n_components). Each element represents the
projection of a sample onto a principal component (PC). Each element PC[i, j] in the
matrix corresponds to how much of the i-th sample contributes to the j-th principal
component.

Loadings matrix (pca.components_) has shape (n_components, n_features). Each element
represents the weight (or "loading") of a feature on a principal component. Each
element loadings[j, k] in this matrix shows the contribution (weight or coefficient)
of the k-th feature to the j-th principal component.

Principal Components Matrix: Use it for data reduction (e.g., reducing the number of
features). Loadings Matrix: Use it to understand how the original features contribute
to each principal component and to interpret the meaning of the principal components
in terms of the original features. It also helps in reconstructing or approximating
the original data from the principal components.

Note: original_data ≈ principal_components × loadings_transposed
"""
# principal components:
principal_components = pca.transform(df_01.T)
df_pc = pd.DataFrame(principal_components, index=tail_ids)
assert principal_components.shape == (4, 4)
pd.DataFrame(np.corrcoef(principal_components, rowvar=False))  # PCs are orthogonal

# loadings:
loadings = pca_01.named_steps["pca"].components_
assert loadings.shape == (4, 9)
pd.DataFrame(np.corrcoef(loadings))  # loadings are not expected to be orthogonal

print(df_01.T)
print(df_pc)

fig = go.Figure(
    data=go.Scatter(x=df_pc[0], y=df_pc[1], text=df_pc.index, mode="markers")
)
fig.show()

# Using tf-idf features:
pca_02 = pca.fit(df_02.T)
pca_02.named_steps["pca"].explained_variance_

# principal components:
principal_components_02 = pca.transform(df_02.T)
pd.DataFrame(principal_components_02)
assert principal_components_02.shape == (4, 4)
pd.DataFrame(np.corrcoef(principal_components_02, rowvar=False))  # PCs are orthogonal

# loadings:
loadings_02 = pca_02.named_steps["pca"].components_
assert loadings_02.shape == (4, 9)
pd.DataFrame(np.corrcoef(loadings_02))  # loadings are not expected to be orthogonal

print(df_02.T)
print(pd.DataFrame(principal_components_02))
