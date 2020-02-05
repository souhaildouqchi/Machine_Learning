# Data processing
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
# Matrix of independent variables
# Matrix of features we take all the lines and all the columns -1 which is the last one
X = dataset.iloc[:, :-1].values
# Matrix of dependant variables
Y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# X_train : training part of the matrix of features y_train : training part of dependant variables
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Taking care of encoding categorical data
# oneHoteEncoder encodes each categories to each column with 1s and 0s
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
# Avoiding the dummy variable trap ( this is done automatically )
# all columns starting from index 1
X = X[:, 1:]
# # features scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Fitting Multiple linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
# recheck here y_pred doest show on variables
y_pred = regressor.predict(X_test)
# building the optimal model using backward elimination
import statsmodels.formula.api as sm

# we add b0 to the statsmodel a vector of 50 ones , as type(int) cast array to int
# ligne axises = 0 colomun = 1
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# X_opt only the highly significant variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.ols(endog = y , exog = X_opt).fit()
regressor_OLS.summary()