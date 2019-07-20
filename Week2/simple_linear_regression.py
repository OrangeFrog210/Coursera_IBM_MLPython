# Date: May 28th, 2019.
# Coursera IBM ML with Python course.

# import needed packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


# reading data in
df = pd.read_csv("/home/yshiba/Downloads/FuelConsumptionCo2.csv")
# taking a look at the dataset
print(df.head)


# Data Exploration
# summarizing the data
print(df.describe())

# selecting some features to explore more
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

# plotting each of the features
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()

# plotting each of the features vs Emission
# in order to see how linear their relation is
# 1. Fuel consumption vs co2 emission
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='yellow')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('Emission')
plt.show()

# 2. Engine size vs co2 emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='green')
plt.xlabel('ENGINESIZE')
plt.ylabel('Emission')
plt.show()

# 3. Cylinder vs co2 emission
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red')
plt.xlabel('CYLINDERS')
plt.ylabel('Emission')
plt.show()


# Splitting our dataset into train and test sets, 80% of the entire data for training
# and the rest for testing
# creating a mask to select random rows using _np.random.rand()_ function

msk = np.random.rand(len(df)) < 0.8
train =cdf[msk]
test =cdf[~msk]
# ~ is used to indicate not


# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='green')
plt.xlabel('ENGINESIZE')
plt.ylabel('Emission')
plt.show()

# Modeling
# using sklearn package to model data
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The following two are the parameter of the fit line.
# coefficient is the slope of the line
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Plotting outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='green')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel('ENGINESIZE')
plt.ylabel('Emission')
plt.show()


# Evaluation
# comparing the actual values and predicted values to calculate the accuracy of a regression model.
# evaluation metrics provide a key role in the development of a model, as it
# provides insight to areas that require improvement.

# Using MSE here
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
