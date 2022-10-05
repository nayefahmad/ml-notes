# # Exploring regression models in sklearn

# ## Overview

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, LassoCV

df = pd.DataFrame(
    {
        "category": ["a", "a", "b", "b"],
        "numeric": [4, 6, 100, 132],
        "target": [12, 24, 360, 420],
    }
)

# Plotting with pandas:

df.plot(x="numeric", y="target", kind="scatter")
plt.show()

sns.boxplot(data=df, x="category", y="target")
plt.show()

# Plotting with mpl:

fig, ax = plt.subplots()
ax.scatter(df.numeric, df.target, label="series_01")
ax.set_title("<title>")
fig.show()

# ## Models
m = LinearRegression()
m2 = Lasso()
m3 = LassoCV()

models = {
    "lm": m,
    "lasso": m2,
    # 'lassoCV': m3
}

# ## Fitting with only categorical variable

X = pd.get_dummies(df["category"])
y = df["target"]

results = {}
for model_name, model in models.items():
    model.fit(X, y)
    coeffs = model.coef_
    train_preds = model.predict(X)
    errors = y.values - train_preds
    results[f"{model_name}"] = (coeffs, train_preds, errors)

group_means = df.groupby("category")["target"].mean().tolist()
# results['lm'][1] in group_means
