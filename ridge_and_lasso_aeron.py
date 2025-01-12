import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
def load_data(filename):
    data = pd.read_excel(filename)
    X = data[['EuroInterestRate', 'USDInterestRate']].values
    y = data['GoldPrice'].values
    date = pd.to_datetime(data['Date'])
    return X, y, date

# Standardize features
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

# Ridge Regression
def ridge_regression(X, y, alpha, max_iter=1000, learning_rate=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for _ in range(max_iter):
        y_pred = np.dot(X, w) + b
        error = y_pred - y

        # Gradients
        dw = (2 / n) * np.dot(X.T, error) + 2 * alpha * w
        db = (2 / n) * np.sum(error)

        # Update weights
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

# Lasso Regression
def lasso_regression(X, y, alpha, max_iter=1000, learning_rate=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for _ in range(max_iter):
        y_pred = np.dot(X, w) + b
        error = y_pred - y

        # Gradients
        dw = (2 / n) * np.dot(X.T, error)
        dw += alpha * np.sign(w)
        db = (2 / n) * np.sum(error)

        # Update weights
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

# Predict function
def predict(X, w, b):
    return np.dot(X, w) + b

# RMSE evaluation
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Main function
if __name__ == "__main__":
    filename = 'Gold_price_Interest_Rates_USD_Euro.xlsx'  # Replace with your data file
    X, y, date = load_data(filename)

    # Standardize features
    X, mean, std = standardize(X)

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    date_test = date[train_size:]  # Corresponding dates for the test set

    # Ridge Regression
    alpha_ridge = 0.01
    w_ridge, b_ridge = ridge_regression(X_train, y_train, alpha=alpha_ridge)
    y_pred_ridge = predict(X_test, w_ridge, b_ridge)
    print(f"Ridge RMSE: {rmse(y_test, y_pred_ridge):.4f}")

    # Lasso Regression
    alpha_lasso = 1.0
    w_lasso, b_lasso = lasso_regression(X_train, y_train, alpha=alpha_lasso)
    y_pred_lasso = predict(X_test, w_lasso, b_lasso)
    print(f"Lasso RMSE: {rmse(y_test, y_pred_lasso):.4f}")

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(date_test, y_test, label="Actual Prices", color='blue')
    plt.plot(date_test, y_pred_ridge, label="Ridge Predictions", color='red', linestyle='--')
    plt.plot(date_test, y_pred_lasso, label="Lasso Predictions", color='green', linestyle='--')
    plt.title("Gold Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
