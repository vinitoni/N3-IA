import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("housing_prices.csv")

print("Informações gerais do dataset:")
print(data.info())
print("\nPrimeiros dados:")
print(data.head())

print("\nNúmero de valores ausentes por coluna:")
print(data.isnull().sum())
data = data.fillna(data.median())  

X = data.drop("SalePrice", axis=1)._get_numeric_data()  
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

lr_rmse = mean_squared_error(y_test, lr_y_pred, squared=False)
lr_r2 = r2_score(y_test, lr_y_pred)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_rmse = mean_squared_error(y_test, rf_y_pred, squared=False)
rf_r2 = r2_score(y_test, rf_y_pred)

print("\nDesempenho da Regressão Linear:")
print(f"RMSE: {lr_rmse}")
print(f"R²: {lr_r2}")

print("\nDesempenho do Random Forest Regressor:")
print(f"RMSE: {rf_rmse}")
print(f"R²: {rf_r2}")
