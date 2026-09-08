# Experiment: Regression Model on Pima Indians Diabetes Dataset

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
df = pd.read_csv("diabetes.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())


# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# 4. Replace invalid zero values with NaN
# Zero is not realistic for these medical attributes
invalid_columns = [
    'glucose_concentration',
    'diastolic_blood_pressure',
    'triceps_sf_thickness',
    'serum_insulin',
    'bmi'
]

for col in invalid_columns:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

print("\nMissing Values After Replacing Invalid Zeros:")
print(df.isnull().sum())


# 5. Fill missing values using median
for col in invalid_columns:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())


# 6. Select features and target
# BMI is the continuous target variable

features = [
    'times_pregnant',
    'glucose_concentration',
    'diastolic_blood_pressure',
    'triceps_sf_thickness',
    'serum_insulin',
    'd_pedigree_function',
    'years_of_age'
]

X = df[features]
y = df['bmi']

print("\nFeatures:")
print(X.head())

print("\nTarget (BMI):")
print(y.head())


# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# 8. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# 9. Generate predictions
y_pred = model.predict(X_test)

print("\nActual vs Predicted BMI:")
comparison = pd.DataFrame({
    'Actual BMI': y_test.values,
    'Predicted BMI': y_pred
})

print(comparison.head(10))


# 10. Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nRegression Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)


# 11. Display model coefficients
print("\nRegression Coefficients:")

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print(coefficients)

print("\nIntercept:", model.intercept_)


# 12. Residual calculation
residuals = y_test - y_pred

print("\nFirst 10 Residuals:")
print(residuals.head(10))


# 13. Residual Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle='--')
plt.xlabel("Predicted BMI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


# 14. Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

# Reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.title("Actual vs Predicted BMI")
plt.show()


# 15. Distribution of residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual")
plt.title("Distribution of Residuals")
plt.show()