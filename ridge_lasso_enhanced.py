import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data with error handling
def load_data(filename):
    try:
        data = pd.read_excel(filename)
        print("File loaded successfully.")
        print("Columns in the file:", data.columns)

        X = data[['EuroInterestRate', 'USDInterestRate']].values
        y = data['GoldPrice'].values
        date = pd.to_datetime(data['Date'])
        return X, y, date
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

# Prediction function
def predict(X, w, b):
    return np.dot(X, w) + b

# Evaluation metrics
def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return rmse, mae, r2

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train, X_test, y_test, regression_func, alpha_values):
    best_alpha = None
    best_rmse = float('inf')

    for alpha in alpha_values:
        w, b = regression_func(X_train, y_train, alpha)
        y_pred = predict(X_test, w, b)
        rmse, _, _ = evaluate_metrics(y_test, y_pred)

        print(f"Alpha: {alpha:.4f}, RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    return best_alpha

# Main function
if __name__ == "__main__":
    filename = 'Gold_Interest_Rates.xlsx'
    try:
        X, y, date = load_data(filename)
    except Exception as e:
        print("Exiting due to file loading error.")
        exit()

    # Standardize features
    X, mean, std = standardize(X)

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    date_test = date[train_size:]  # Corresponding dates for the test set

    # Hyperparameter tuning
    alpha_values = np.logspace(-3, 1, 10)  # Test values for alpha

    print("\nTuning Ridge Regression...")
    best_alpha_ridge = tune_hyperparameters(X_train, y_train, X_test, y_test, ridge_regression, alpha_values)
    print(f"Best Ridge Alpha: {best_alpha_ridge}")

    print("\nTuning Lasso Regression...")
    best_alpha_lasso = tune_hyperparameters(X_train, y_train, X_test, y_test, lasso_regression, alpha_values)
    print(f"Best Lasso Alpha: {best_alpha_lasso}")

    # Train final models with best alpha
    w_ridge, b_ridge = ridge_regression(X_train, y_train, best_alpha_ridge)
    y_pred_ridge = predict(X_test, w_ridge, b_ridge)

    w_lasso, b_lasso = lasso_regression(X_train, y_train, best_alpha_lasso)
    y_pred_lasso = predict(X_test, w_lasso, b_lasso)

    # Evaluate final models
    print("\nFinal Model Performance:")
    ridge_metrics = evaluate_metrics(y_test, y_pred_ridge)
    lasso_metrics = evaluate_metrics(y_test, y_pred_lasso)

    print(f"Ridge - RMSE: {ridge_metrics[0]:.4f}, MAE: {ridge_metrics[1]:.4f}, R²: {ridge_metrics[2]:.4f}")
    print(f"Lasso - RMSE: {lasso_metrics[0]:.4f}, MAE: {lasso_metrics[1]:.4f}, R²: {lasso_metrics[2]:.4f}")

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
