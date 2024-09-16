"""
# Exploring TF-IDF

Reference: V. Boykis, "Embeddings" doc.

"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
pca = PCA(random_state=seed, n_components=4)
pca_01 = pca.fit(df_01)

pca_01.components_
# assert pca_01.components_.shape == (4,9)
pca_01.explained_variance_

df_01_pca = pd.DataFrame(
    pca_01.components_,  # index=tail_ids # , columns=vectorizer_01.get_feature_names()
).T

print(df_01)
print(df_01_pca)


# Using tf-idf features:
pca = PCA(random_state=seed, n_components=1)
pca_02 = pca.fit(df_02.T)

pca_02.components_
# assert pca_02.components_.shape == (4,9)
pca_02.explained_variance_

df_02_pca = pd.DataFrame(
    pca_02.components_, index=tail_ids  # , columns=vectorizer_02.get_feature_names()
).T

print(df_02)
print(df_02_pca)
