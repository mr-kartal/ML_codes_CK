import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load data function
def load_data(filename):
    data = pd.read_excel(filename)
    dates = np.arange(len(data))
    prices = data['Kapanış'].values
    return dates, prices


# Simple Linear Regression function
def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - m * x_mean
    return m, b


# Polynomial Regression function
def polynomial_regression(x, y, degree):
    coefficients = np.polyfit(x, y, degree)
    return coefficients


# Unified function for both tasks
def train_and_plot(dates, prices, period, regression_type='linear', degree=2):
    models = []  # Store models (linear or polynomial)

    # Split data into periods and train models
    for i in range(0, len(dates), period):
        x = dates[i:i + period]
        y = prices[i:i + period]
        if len(x) < period:  # Skip if there are not enough points for the last segment
            break

        if regression_type == 'linear':
            # Train simple linear regression
            m, b = linear_regression(x, y)
            models.append((m, b, x))
        elif regression_type == 'polynomial':
            # Train polynomial regression
            coefficients = polynomial_regression(x, y, degree)
            models.append((coefficients, x))
        else:
            raise ValueError("Invalid regression_type. Use 'linear' or 'polynomial'.")

    # Plot actual data
    plt.figure(figsize=(12, 6))
    plt.scatter(dates, prices, color='blue', label='Actual Prices', s=10)

    # Plot regression lines or curves
    for model in models:
        x = model[-1]
        if regression_type == 'linear':
            m, b = model[0], model[1]
            y_pred = m * x + b
            plt.plot(x, y_pred, color='red', label=f'Linear (Period {len(x)})')
        elif regression_type == 'polynomial':
            coefficients = model[0]
            y_pred = np.polyval(coefficients, x)
            plt.plot(x, y_pred, label=f'Polynomial Degree {len(coefficients) - 1}')

    # Add title and labels
    plt.title(f'{regression_type.capitalize()} Regression for Gold Prices')
    plt.xlabel('Days')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    # Load the dataset
    filename = 'gold_price_closed.xlsx'
    dates, prices = load_data(filename)


    print("Running Simple Linear Regression (Task 1)")
    period = 30  # Set the time period (30 days, 60 days, etc.)
    train_and_plot(dates, prices, period, regression_type='linear')

    print("Running Polynomial Regression (Task 2)")
    degree = 3  # Set the polynomial degree (e.g., 2, 3, ..., 8)
    train_and_plot(dates, prices, period, regression_type='polynomial', degree=degree)
