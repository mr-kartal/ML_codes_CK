import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data function with debugging
def load_data(filename):
    try:
        # Load Excel file
        data = pd.read_excel(filename)
        print("File loaded successfully.")
        print("Columns in the file:", data.columns)

        # Ensure column names match the dataset
        X = data[['EuroInterestRate', 'USDInterestRate']].values
        y = data['GoldPrice'].values
        return X, y
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

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
    filename = 'Gold_Interest_Rates.xlsx'  # Replace with your file name
    try:
        X, y = load_data(filename)
        print("Data shape:", X.shape, y.shape)
    except Exception as e:
        print("Exiting due to file loading error.")

    # Standardize features
    X, mean, std = standardize(X)

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

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
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Prices", color='blue')
    plt.plot(y_pred_ridge, label="Ridge Predictions", color='red', linestyle='--')
    plt.plot(y_pred_lasso, label="Lasso Predictions", color='green', linestyle='--')
    plt.title("Gold Price Prediction")
    plt.xlabel("Test Samples")
    plt.ylabel("Gold Price")
    plt.legend()
    plt.show()
