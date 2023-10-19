""" Compute regression baseline by using the training data mean value as the prediction for all test
samples.
Print baseline MSE.
Note: You can use MSE available in the sklearn.metrics sub-package.
(b) Linear model 
Use sklearn.linear model to fit a linear model to the data.
Use the linear model to predict the values for the test data and print the test set MSE.
(c) Decision tree regressor 
Use sklearn.tree.DecisionTreeRegressor to fit a decision tree to your data.
Use the model to predict the values for the test data and print the test set MSE.
(d) Random forest regressor 
Use sklearn.ensemble.RandomForestRegressor to fit a random forest to your data.
Use the model to predict the values for the test data and print the test set MSE. """


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the training and test data
X_train = np.loadtxt('disease/disease_X_train.txt')
X_test = np.loadtxt('disease/disease_X_test.txt')
y_train = np.loadtxt('disease/disease_y_train.txt')
y_test = np.loadtxt('disease/disease_y_test.txt')

# Compute the mean of y_train as the baseline prediction
baseline_prediction = np.mean(y_train)

# Calculate MSE for the baseline prediction
baseline_mse = mean_squared_error(y_test, np.full_like(y_test, baseline_prediction))
print(f"Baseline MSE: {baseline_mse}")


# Create and fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict the values for the test data
linear_predictions = linear_model.predict(X_test)
print(f"Predicted values for the test data: {linear_predictions}")

# Calculate MSE for the linear model
linear_mse = mean_squared_error(y_test, linear_predictions)
print(f"Linear Model MSE: {linear_mse}")


# Create and fit a decision tree regressor
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(X_train, y_train)

# Predict the values for the test data using the decision tree model
decision_tree_predictions = decision_tree_model.predict(X_test)

# Calculate MSE for the decision tree model
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)
print(f"Decision Tree Regressor MSE: {decision_tree_mse}")


# Create and fit a random forest regressor
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

# Predict the values for the test data using the random forest model
random_forest_predictions = random_forest_model.predict(X_test)

# Calculate MSE for the random forest model
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)
print(f"Random Forest Regressor MSE: {random_forest_mse}")
