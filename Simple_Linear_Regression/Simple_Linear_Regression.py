# Data processing
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
# Matrix of independent variables
# Matrix of features we take all the lines and all the columns -1 which is the last one
X = dataset.iloc[:, :-1].values
# Matrix of dependant variables
Y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# X_train : training part of the matrix of features y_train : training part of dependant variables
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# Fiting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

# we dont need to put any parameters in the instance of the class
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the training set Result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# its the end of the graph and we need to plot it
plt.show()
# Visualising the test set Result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# its the end of the graph and we need to plot it
plt.show()
# # features scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
