import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Define base models
base_models = [
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
]

# Train base models and generate predictions for the validation set
val_predictions = np.zeros((X_val.shape[0], len(base_models)))
test_predictions = np.zeros((X_test.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_train, y_train)  # Train each base model on the training data
    val_predictions[:, i] = model.predict(X_val)  # Predict on the validation data
    test_predictions[:, i] = model.predict(X_test)  # Predict on the test data

# Train the meta-model using the validation set predictions
meta_model = LinearRegression()
meta_model.fit(val_predictions, y_val)

# Make final predictions on the test set
final_predictions = meta_model.predict(test_predictions)

# Evaluate the performance
rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(f"Stacked Model RMSE: {rmse:.4f}")
