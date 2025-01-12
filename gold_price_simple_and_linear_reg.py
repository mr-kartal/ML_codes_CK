import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Step 1: Read the dataset
def load_data(filename):
    data = pd.read_excel(filename)
    dates = np.arange(len(data))  # Convert dates to sequential integers for simplicity
    prices = data['Kapanış'].values
    return dates, prices



# Task 1: Simple Linear Regression from Scratch
def linear_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - m * x_mean
    return m, b


def train_linear_models(dates, prices, period):
    regression_lines = []
    for i in range(0, len(dates), period):
        x = dates[i:i + period]
        y = prices[i:i + period]
        if len(x) < period:
            break
        m, b = linear_regression(x, y)
        regression_lines.append((m, b, x))
    return regression_lines


def plot_linear_regression(dates, prices, regression_lines):
    plt.figure(figsize=(12, 6))
    plt.scatter(dates, prices, color='blue', label='Actual Prices', s=10)

    for m, b, x in regression_lines:
        y_pred = m * x + b
        plt.plot(x, y_pred, color='red', label=f'Linear Regression (Period {len(x)})')

    plt.title('Simple Linear Regression for Gold Prices')
    plt.xlabel('Days')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.show()


# Task 2: Polynomial Regression from Scratch
def polynomial_regression(x, y, degree):
    coefficients = np.polyfit(x, y, degree)
    return coefficients


def train_polynomial_models(dates, prices, period, degree):
    polynomial_curves = []
    for i in range(0, len(dates), period):
        x = dates[i:i + period]
        y = prices[i:i + period]
        if len(x) < period:
            break
        coefficients = polynomial_regression(x, y, degree)
        polynomial_curves.append((coefficients, x))
    return polynomial_curves


def plot_polynomial_regression(dates, prices, polynomial_curves):
    plt.figure(figsize=(12, 6))
    plt.scatter(dates, prices, color='blue', label='Actual Prices', s=10)

    for coefficients, x in polynomial_curves:
        y_pred = np.polyval(coefficients, x)
        plt.plot(x, y_pred, label=f'Polynomial Degree {len(coefficients) - 1}')

    plt.title('Polynomial Regression for Gold Prices')
    plt.xlabel('Days')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    # Load the dataset
    filename = 'gold_price_closed.xlsx'
    dates, prices = load_data(filename)

    # TASK 1: Simple Linear Regression
    print("Running TASK 1: Simple Linear Regression")
    period = 30  # Set the time period (30 days, 60 days, etc.)
    regression_lines = train_linear_models(dates, prices, period)
    plot_linear_regression(dates, prices, regression_lines)

    # TASK 2: Polynomial Regression
    print("Running TASK 2: Polynomial Regression")
    degree = 3  # Set the polynomial degree (e.g., 2, 3, ..., 8)
    polynomial_curves = train_polynomial_models(dates, prices, period, degree)
    plot_polynomial_regression(dates, prices, polynomial_curves)
