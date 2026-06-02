"""Generate a realistic synthetic Brent Oil Prices dataset (1987-2023)."""
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate business days from 1987 to 2023
dates = pd.bdate_range(start='1987-05-20', end='2023-12-29')

n = len(dates)
prices = np.zeros(n)
prices[0] = 18.63  # Starting price in 1987

# Define regime parameters to mimic real Brent oil behavior
# Each regime: (start_date, end_date, drift, volatility)
regimes = [
    ('1987-05-20', '1990-07-01', 0.0001, 0.018),   # Stable low period
    ('1990-07-02', '1991-03-01', 0.002, 0.035),     # Gulf War spike
    ('1991-03-02', '1997-12-31', 0.0001, 0.016),    # Post-war stability
    ('1998-01-01', '1999-03-01', -0.001, 0.025),    # Asian crisis dip
    ('1999-03-02', '2000-09-01', 0.0015, 0.022),    # Recovery
    ('2000-09-02', '2001-12-31', -0.0005, 0.022),   # Dot-com/9-11
    ('2002-01-01', '2008-07-01', 0.0008, 0.018),    # China boom / super-cycle
    ('2008-07-02', '2009-02-01', -0.004, 0.045),    # Financial crisis crash
    ('2009-02-02', '2011-04-01', 0.0012, 0.022),    # Recovery to 2011
    ('2011-04-02', '2014-06-01', 0.0001, 0.012),    # High plateau ~$110
    ('2014-06-02', '2016-01-15', -0.002, 0.030),    # OPEC price war crash
    ('2016-01-16', '2020-01-01', 0.0005, 0.018),    # Gradual recovery
    ('2020-01-02', '2020-04-20', -0.006, 0.060),    # COVID crash
    ('2020-04-21', '2022-03-01', 0.002, 0.025),     # Post-COVID recovery
    ('2022-03-02', '2022-06-30', 0.0005, 0.030),    # Ukraine war spike
    ('2022-07-01', '2023-12-29', -0.0003, 0.016),   # Normalization
]

# Target approximate price levels for key dates
target_prices = {
    '1990-10-01': 40,
    '1991-04-01': 20,
    '1998-12-01': 10,
    '2000-09-01': 35,
    '2001-11-01': 20,
    '2008-07-01': 145,
    '2009-02-01': 40,
    '2011-04-01': 125,
    '2014-06-01': 115,
    '2016-01-15': 28,
    '2020-01-01': 66,
    '2020-04-20': 20,
    '2022-03-01': 130,
    '2023-12-29': 77,
}

# Build price series using geometric Brownian motion with regime-dependent parameters
for i in range(1, n):
    current_date = dates[i]

    # Find current regime
    drift = 0.0001
    vol = 0.018
    for r_start, r_end, r_drift, r_vol in regimes:
        if pd.Timestamp(r_start) <= current_date <= pd.Timestamp(r_end):
            drift = r_drift
            vol = r_vol
            break

    # GBM step
    shock = np.random.normal(0, 1)
    prices[i] = prices[i-1] * np.exp((drift - 0.5 * vol**2) + vol * shock)

    # Ensure price stays positive and reasonable
    prices[i] = max(prices[i], 5.0)
    prices[i] = min(prices[i], 200.0)

# Apply gentle nudging toward target prices to make the series realistic
# Use a second pass with exponential smoothing toward targets
for target_date_str, target_price in target_prices.items():
    target_date = pd.Timestamp(target_date_str)
    idx = np.searchsorted(dates, target_date)
    if idx >= n:
        idx = n - 1

    # Get current price at target date
    current_at_target = prices[idx]
    if current_at_target <= 0:
        continue

    # Calculate adjustment ratio
    ratio = target_price / current_at_target

    # Apply gradual adjustment over 60 trading days before the target
    window = min(60, idx)
    for j in range(idx - window, idx + 1):
        if j < 0:
            continue
        # Linear interpolation of the adjustment
        t = (j - (idx - window)) / window
        adjustment = 1.0 + (ratio - 1.0) * t
        prices[j] *= adjustment

# Final smoothing pass to remove discontinuities
for i in range(1, n):
    # Slight mean reversion to prevent extreme values
    if prices[i] > 180:
        prices[i] = prices[i] * 0.98
    if prices[i] < 8:
        prices[i] = prices[i] * 1.02

# Round to 2 decimal places
prices = np.round(prices, 2)

# Create DataFrame
df = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),
    'Price': prices
})

# Save to CSV
df.to_csv('data/BrentOilPrices.csv', index=False)
print(f"Generated {len(df)} rows of synthetic Brent oil price data")
print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
print(f"Final price: ${df['Price'].iloc[-1]:.2f}")
