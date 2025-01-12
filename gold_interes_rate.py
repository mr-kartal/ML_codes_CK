import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Load data function
def load_data(filename):
    data = pd.read_excel(filename, sheet_name='Data')
    # Ensure the data is sorted by date
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')
    # Extract features and target variable
    X = data[['EuroInterestRate', 'USDInterestRate']].values
    y = data['GoldPrice'].values
    return X, y, data['Date']


# Simple Linear Regression function
def linear_regression(X, y):
    # Add a column of ones to X for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Calculate weights using the Normal Equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best


# Polynomial Regression function
def polynomial_regression(X, y, degree):
    from itertools import combinations_with_replacement
    # Generate polynomial features
    n_samples, n_features = X.shape
    combinations = [combinations_with_replacement(range(n_features), i) for i in range(1, degree + 1)]
    combinations = [item for sublist in combinations for item in sublist]
    X_poly = np.ones((n_samples, len(combinations) + 1))
    for i, comb in enumerate(combinations):
        X_poly[:, i + 1] = np.prod(X[:, comb], axis=1)
    # Calculate weights using the Normal Equation
    theta_best = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
    return theta_best, X_poly


# Prediction function
def predict(X, theta, degree=1):
    if degree == 1:
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(theta)
    else:
        from itertools import combinations_with_replacement
        n_samples, n_features = X.shape
        combinations = [combinations_with_replacement(range(n_features), i) for i in range(1, degree + 1)]
        combinations = [item for sublist in combinations for item in sublist]
        X_poly = np.ones((n_samples, len(combinations) + 1))
        for i, comb in enumerate(combinations):
            X_poly[:, i + 1] = np.prod(X[:, comb], axis=1)
        return X_poly.dot(theta)


# Plotting function
def plot_predictions(dates, actual_prices, predicted_prices, title):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, label='Actual Gold Prices', color='blue')
    plt.plot(dates, predicted_prices, label='Predicted Gold Prices', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Gold Price (USD)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function
if __name__ == "__main__":
    # Load the dataset
    filename = 'Gold_Interest_Rates.xlsx'  # Replace with your dataset file
    X, y, dates = load_data(filename)

    # TASK 1: Simple Linear Regression
    print("Running Simple Linear Regression")
    theta_linear = linear_regression(X, y)
    y_pred_linear = predict(X, theta_linear)
    plot_predictions(dates, y, y_pred_linear, 'Simple Linear Regression: Gold Price Prediction')

    # TASK 2: Polynomial Regression
    print("Running Polynomial Regression")
    degree = 2  # Set the polynomial degree
    theta_poly, X_poly = polynomial_regression(X, y, degree)
    y_pred_poly = predict(X, theta_poly, degree)
    plot_predictions(dates, y, y_pred_poly, f'Polynomial Regression (Degree {degree}): Gold Price Prediction')
