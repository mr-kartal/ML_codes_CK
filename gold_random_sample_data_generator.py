import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
start_date = datetime(2020, 1, 1)
num_days = 500  # Number of days to simulate
dates = [start_date + timedelta(days=i) for i in range(num_days)]

# Simulate realistic gold prices (trending upward with small fluctuations)
gold_prices = np.linspace(1500, 2000, num_days) + np.random.normal(0, 20, num_days)

# Simulate Euro interest rates (gradual increase with small noise)
euro_rates = np.linspace(-0.5, 0.2, num_days) + np.random.normal(0, 0.02, num_days)

# Simulate USD interest rates (gradual increase with small noise)
usd_rates = np.linspace(0.25, 1.0, num_days) + np.random.normal(0, 0.02, num_days)

# Create a DataFrame
data = {
    'Date': dates,
    'GoldPrice': gold_prices,
    'EuroInterestRate': euro_rates,
    'USDInterestRate': usd_rates
}
df = pd.DataFrame(data)

# Save to Excel
filename = 'Gold_Interest_Rates.xlsx'
df.to_excel(filename, index=False, sheet_name='Data')
print(f"Excel file '{filename}' with 500 rows of data created successfully.")
