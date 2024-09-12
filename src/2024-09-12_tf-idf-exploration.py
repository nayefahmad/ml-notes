"""
# Exploring TF-IDF

Reference: V. Boykis, "Embeddings" doc.

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Represent two tails by their history of destinations
tails = ["YUL, YYZ, YUL, YYZ, JFK", "JFK, LAX, JFK, LAX, YYZ, YVR, JFK"]

vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(tails)
assert vector.shape == (2, 5)

dict(zip(vectorizer.get_feature_names(), vector.toarray()))

pd.DataFrame(
    vector.toarray(), index=["tail01", "tail02"], columns=vectorizer.get_feature_names()
).T
