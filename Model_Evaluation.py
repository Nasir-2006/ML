import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv('BostonHousing.csv')
df = df.dropna()
print(f"Total traning examples {len(df)}")
x = df[['lstat']].values
y = df[['medv']].values
x_train,x_t,y_train,y_t = train_test_split(x,y, test_size = 0.40, random_state=1)
x_cv,x_test,y_cv,y_test = train_test_split(x_t,y_t, test_size= 0.50,random_state=1)
scalar1 = StandardScaler()
x_train_sc = scalar1.fit_transform(x_train)
x_cv_sc = scalar1.transform(x_cv)
x_test_sc = scalar1.transform(x_test)
Nasir = LinearRegression()
Nasir.fit(x_train_sc,y_train)
x_cv_pre = Nasir.predict(x_cv_sc)
MSE1 = mean_squared_error(y_cv,x_cv_pre)/2
print(f"MSE1 is {MSE1}")
poly = PolynomialFeatures(degree = 2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_cv_poly = poly.transform(x_cv)
x_test_poly = poly.transform(x_test)
scalar2 = StandardScaler()
x_train_poly_sc = scalar2.fit_transform(x_train_poly)
x_cv_poly_sc = scalar2.transform(x_cv_poly)
x_test_poly_sc = scalar2.transform(x_test_poly)
MUBASHER = LinearRegression()
MUBASHER.fit(x_train_poly_sc,y_train)
x_cv_pol_pre = MUBASHER.predict(x_cv_poly_sc)
MSE2 = mean_squared_error(y_cv,x_cv_pol_pre)/2
print(f"MSE2 is {MSE2}")

