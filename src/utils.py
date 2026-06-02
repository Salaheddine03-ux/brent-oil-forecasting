"""Utility functions for Brent Oil Price Time Series Analysis."""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def load_data(filepath='data/BrentOilPrices.csv'):
    """Load Brent oil prices data and parse dates."""
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    return df


def plot_time_series(series, title='Brent Oil Prices', ylabel='Price (USD)',
                     xlabel='Date', figsize=(14, 6), save_path=None):
    """Plot a time series."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, linewidth=0.8, color='#2196F3')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def check_stationarity(series, series_name='Series'):
    """Run ADF and KPSS tests for stationarity."""
    results = {}

    # ADF Test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['adf'] = {
        'test_statistic': adf_result[0],
        'p_value': adf_result[1],
        'lags_used': adf_result[2],
        'n_obs': adf_result[3],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }

    # KPSS Test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    results['kpss'] = {
        'test_statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'lags_used': kpss_result[2],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] > 0.05
    }

    print(f"\n{'='*60}")
    print(f"Stationarity Tests for: {series_name}")
    print(f"{'='*60}")
    print(f"ADF Test Statistic: {adf_result[0]:.4f}")
    print(f"ADF p-value: {adf_result[1]:.6f}")
    print(f"ADF Stationary: {'Yes' if results['adf']['is_stationary'] else 'No'}")
    print(f"KPSS Test Statistic: {kpss_result[0]:.4f}")
    print(f"KPSS p-value: {kpss_result[1]:.4f}")
    print(f"KPSS Stationary: {'Yes' if results['kpss']['is_stationary'] else 'No'}")
    print(f"{'='*60}")

    return results


def evaluate_forecast(actual, predicted, model_name='Model'):
    """Compute MAE, RMSE, and MAPE for forecast evaluation."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE with protection against zero values
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan

    results = {
        'model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4)
    }

    print(f"{model_name:30s} | MAE: {mae:8.4f} | RMSE: {rmse:8.4f} | MAPE: {mape:6.2f}%")
    return results


def create_lag_features(df, target_col='Price', lags=[1, 7, 14, 30],
                        rolling_windows=[7, 14, 30]):
    """Create lag and rolling features for ML models."""
    result = df.copy()

    # Lag features
    for lag in lags:
        result[f'lag_{lag}'] = result[target_col].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        result[f'rolling_mean_{window}'] = result[target_col].shift(1).rolling(window=window).mean()
        result[f'rolling_std_{window}'] = result[target_col].shift(1).rolling(window=window).std()

    # Calendar features
    if isinstance(result.index, pd.DatetimeIndex):
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['day_of_week'] = result.index.dayofweek
        result['year'] = result.index.year

    return result


def plot_acf_pacf(series, lags=50, title_prefix='', save_path=None):
    """Plot ACF and PACF for a given series."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(series.dropna(), lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'{title_prefix} ACF', fontsize=12)

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'{title_prefix} PACF', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_forecast(train, test, forecast, title='Forecast', ylabel='Price (USD)',
                  save_path=None):
    """Plot train, test, and forecast data."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index, train.values, label='Train', color='#2196F3', linewidth=0.8)
    ax.plot(test.index, test.values, label='Test (Actual)', color='#4CAF50', linewidth=1.2)
    ax.plot(test.index[:len(forecast)], forecast[:len(test)],
            label='Forecast', color='#F44336', linewidth=1.2, linestyle='--')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig
