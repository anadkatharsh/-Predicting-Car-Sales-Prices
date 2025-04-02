
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'C:/Users/vtpha/OneDrive/Documents/NEU Courses/Data Mining/Group Project/car_prices.csv'
original_df = pd.read_csv(file_path)

# Trim dataset down to 50000 for more managable analysis
df = original_df.sample(n=50000, random_state=0)

# Dropping non-significant columns to analysis
df.drop(columns=['saledate', 'vin', 'seller'], inplace=True)

# Dropping rows with missing values for 'odometer', 'sellingprice', and 'condition'
df.dropna(subset=['odometer', 'sellingprice', 'condition'], inplace=True)

# Fill in missing values of categorical variables with most frequent/mode
categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'color', 'interior']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Visualizations of distribution of variables in dataset
num_cols = df.select_dtypes(include=[np.number]).columns

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Setting 'sellingprice' as target variable
X = df.drop(columns=['sellingprice'])
y = df['sellingprice']

# Encode categorical variables for use in model analysis
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'Linear Regression MSE: {mse_lr}')
print(f'Linear Regression R2: {r2_lr}')

# Random Forest Model
rf = RandomForestRegressor(n_estimators=50, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest MSE: {mse_rf}')
print(f'Random Forest R2: {r2_rf}')

# Show variables and coefficients for Linear Regression model
coefficients = lr.coef_
feature_names = X_train.columns

coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print("Linear Regression Coefficients:")
print(coeff_df)

# Show variables and coefficients for Random Forest model
importances = rf.feature_importances_
feature_names = X_train.columns

importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

print("Random Forest Feature Importances:")
print(importances_df)