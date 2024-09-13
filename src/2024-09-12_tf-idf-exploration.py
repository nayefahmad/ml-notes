"""
# Exploring TF-IDF

Reference: V. Boykis, "Embeddings" doc.

"""
import pandas as pd
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
