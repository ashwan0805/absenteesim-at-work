import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv('Absenteeism_at_work.csv', delimiter=';')

# Separate features (X) and target (y)
X = df.drop(columns=["Absenteeism time in hours", "ID"])  # Drop target and ID (irrelevant for modeling)
y = df["Absenteeism time in hours"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
decision_tree = DecisionTreeRegressor(random_state=42)
random_forest = RandomForestRegressor(random_state=42)

# Train models
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Make predictions
y_pred_tree = decision_tree.predict(X_test)
y_pred_forest = random_forest.predict(X_test)

# Evaluate performance
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)

mae_forest = mean_absolute_error(y_test, y_pred_forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)

# Print results
print(f"Decision Tree - MAE: {mae_tree:.2f}, MSE: {mse_tree:.2f}")
print(f"Random Forest - MAE: {mae_forest:.2f}, MSE: {mse_forest:.2f}")

# Plot feature importances for Random Forest
importances = random_forest.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices], palette="coolwarm")
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Scatter plot to compare predicted vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_forest, color='blue', label='Random Forest Predictions')
plt.scatter(y_test, y_pred_tree, color='red', label='Decision Tree Predictions', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Absenteeism Hours")
plt.ylabel("Predicted Absenteeism Hours")
plt.legend()
plt.title("Actual vs Predicted Absenteeism Hours")
plt.tight_layout()
plt.show()
