"""
Brent Oil Price Time Series Analysis and Forecasting
Complete pipeline: 11 analysis steps with figure generation.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pmdarima as pm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (load_data, check_stationarity, evaluate_forecast,
                       create_lag_features, plot_acf_pacf)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================================
# STEP 1: Load and Visualize Data
# ============================================================================
def step1_visualization():
    print("\n" + "="*70)
    print("STEP 1: Load and Visualize Data")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step1_visualization')
    ensure_dir(save_dir)

    df = load_data(os.path.join(DATA_DIR, 'BrentOilPrices.csv'))
    print(f"Dataset loaded: {len(df)} observations")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")

    # 1a. Full time series plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Price'], linewidth=0.7, color='#1976D2')
    ax.set_title('Prix du Brent (1987-2023)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix (USD/baril)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.fill_between(df.index, df['Price'], alpha=0.15, color='#1976D2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 1b. Yearly averages
    yearly_avg = df['Price'].resample('YE').mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(yearly_avg.index.year, yearly_avg.values, color='#42A5F5', edgecolor='#1565C0')
    ax.set_title('Moyenne annuelle du prix du Brent', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annee', fontsize=12)
    ax.set_ylabel('Prix moyen (USD/baril)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'yearly_averages.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 1c. Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, bins=50, color='#42A5F5', ax=ax)
    ax.axvline(df['Price'].mean(), color='red', linestyle='--', label=f"Moyenne: ${df['Price'].mean():.2f}")
    ax.axvline(df['Price'].median(), color='green', linestyle='--', label=f"Mediane: ${df['Price'].median():.2f}")
    ax.set_title('Distribution des prix du Brent', fontsize=14, fontweight='bold')
    ax.set_xlabel('Prix (USD/baril)', fontsize=12)
    ax.set_ylabel('Frequence', fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 1 figures saved.")
    return df


# ============================================================================
# STEP 2: Time Series Decomposition
# ============================================================================
def step2_decomposition(df):
    print("\n" + "="*70)
    print("STEP 2: Time Series Decomposition")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step2_decomposition')
    ensure_dir(save_dir)

    # Resample to monthly
    monthly = df['Price'].resample('MS').mean()

    # Additive decomposition
    decomposition = seasonal_decompose(monthly, model='additive', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    axes[0].plot(decomposition.observed, color='#1976D2')
    axes[0].set_title('Observe', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Prix')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(decomposition.trend, color='#F57C00')
    axes[1].set_title('Tendance', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Prix')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(decomposition.seasonal, color='#388E3C')
    axes[2].set_title('Saisonnalite', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Prix')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(decomposition.resid, color='#D32F2F')
    axes[3].set_title('Residu', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Prix')
    axes[3].grid(True, alpha=0.3)

    plt.suptitle('Decomposition de la serie temporelle (Additive, periode=12)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 2 figures saved.")
    return monthly


# ============================================================================
# STEP 3: Stationarity Check
# ============================================================================
def step3_stationarity(df):
    print("\n" + "="*70)
    print("STEP 3: Stationarity Check")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step3_stationarity')
    ensure_dir(save_dir)

    series = df['Price']

    # Rolling statistics (252 trading days ~ 1 year)
    window = 252
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(series.index, series.values, label='Original', color='#1976D2', alpha=0.6, linewidth=0.7)
    axes[0].plot(rolling_mean.index, rolling_mean.values, label=f'Moyenne mobile ({window}j)',
                 color='#D32F2F', linewidth=2)
    axes[0].set_title('Serie originale et moyenne mobile', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Prix (USD)')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling_std.index, rolling_std.values, label=f'Ecart-type mobile ({window}j)',
                 color='#F57C00', linewidth=2)
    axes[1].set_title('Ecart-type mobile', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Ecart-type')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rolling_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Statistical tests
    check_stationarity(series, 'Prix Brent (Original)')

    print("Step 3 figures saved.")


# ============================================================================
# STEP 4: Transformations
# ============================================================================
def step4_transformations(df):
    print("\n" + "="*70)
    print("STEP 4: Transformations")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step4_transformations')
    ensure_dir(save_dir)

    series = df['Price']

    # Log transformation
    log_series = np.log(series)

    # First differencing of log series
    log_diff = log_series.diff().dropna()

    # Box-Cox transformation
    boxcox_series, lam = stats.boxcox(series[series > 0])
    print(f"Box-Cox lambda: {lam:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(series.index, series.values, color='#1976D2', linewidth=0.7)
    axes[0, 0].set_title('Serie originale', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(log_series.index, log_series.values, color='#388E3C', linewidth=0.7)
    axes[0, 1].set_title('Transformation logarithmique', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(log_diff.index, log_diff.values, color='#F57C00', linewidth=0.7)
    axes[1, 0].set_title('Differenciation (log)', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(series.index, boxcox_series, color='#D32F2F', linewidth=0.7)
    axes[1, 1].set_title(f'Transformation Box-Cox (lambda={lam:.3f})', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Comparaison des transformations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'transformations.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Stationarity tests on transformed series
    check_stationarity(log_series, 'Log(Prix)')
    check_stationarity(log_diff, 'Diff(Log(Prix))')

    print("Step 4 figures saved.")
    return log_series


# ============================================================================
# STEP 5: Train/Test Split
# ============================================================================
def step5_train_test_split(df, monthly):
    print("\n" + "="*70)
    print("STEP 5: Train/Test Split")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step5_train_test_split')
    ensure_dir(save_dir)

    # 80/20 split for monthly data
    n = len(monthly)
    split_idx = int(n * 0.8)
    train = monthly[:split_idx]
    test = monthly[split_idx:]

    print(f"Total observations (monthly): {n}")
    print(f"Training set: {len(train)} ({len(train)/n*100:.1f}%)")
    print(f"Test set: {len(test)} ({len(test)/n*100:.1f}%)")
    print(f"Split date: {monthly.index[split_idx].strftime('%Y-%m-%d')}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index, train.values, label='Train (80%)', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values, label='Test (20%)', color='#D32F2F', linewidth=1.2)
    ax.axvline(x=monthly.index[split_idx], color='black', linestyle='--', linewidth=2,
               label=f'Split: {monthly.index[split_idx].strftime("%Y-%m")}')
    ax.set_title('Division Train/Test (80/20)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix (USD/baril)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_test_split.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 5 figures saved.")
    return train, test


# ============================================================================
# STEP 6: Naive Forecasting
# ============================================================================
def step6_naive_forecasting(train, test):
    print("\n" + "="*70)
    print("STEP 6: Naive Forecasting")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step6_naive_forecasting')
    ensure_dir(save_dir)

    results = []
    n_test = len(test)

    # 1. Last Value Naive
    naive_last = np.full(n_test, train.iloc[-1])
    results.append(evaluate_forecast(test.values, naive_last, 'Last Value Naive'))

    # 2. Seasonal Naive (12 months)
    seasonal_period = 12
    seasonal_naive = np.array([train.iloc[-(seasonal_period - i % seasonal_period)]
                               for i in range(n_test)])
    results.append(evaluate_forecast(test.values, seasonal_naive, 'Seasonal Naive (12)'))

    # 3. Moving Average (12 months)
    ma_window = 12
    ma_value = train.iloc[-ma_window:].mean()
    ma_forecast = np.full(n_test, ma_value)
    results.append(evaluate_forecast(test.values, ma_forecast, 'Moving Average (12)'))

    # 4. Simple Exponential Smoothing
    ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.3, optimized=False)
    ses_forecast = ses_model.forecast(n_test)
    results.append(evaluate_forecast(test.values, ses_forecast.values, 'Simple Exp. Smoothing'))

    # Log-transformed naive forecasts
    log_train = np.log(train)
    log_test = np.log(test)

    log_naive_last = np.full(n_test, log_train.iloc[-1])
    log_forecast_back = np.exp(log_naive_last)
    results.append(evaluate_forecast(test.values, log_forecast_back, 'Log Naive (Last Value)'))

    # Plot all forecasts
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    forecasts_to_plot = [
        (naive_last, 'Last Value Naive', '#D32F2F'),
        (seasonal_naive, 'Seasonal Naive (12)', '#F57C00'),
        (ma_forecast, 'Moving Average (12)', '#388E3C'),
        (ses_forecast.values, 'Simple Exp. Smoothing', '#7B1FA2'),
    ]

    for idx, (forecast, name, color) in enumerate(forecasts_to_plot):
        ax = axes[idx // 2, idx % 2]
        ax.plot(train.index[-36:], train.values[-36:], label='Train', color='#1976D2', linewidth=1)
        ax.plot(test.index, test.values, label='Actual', color='#4CAF50', linewidth=1.5)
        ax.plot(test.index, forecast[:n_test], label=name, color=color, linewidth=1.5, linestyle='--')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Previsions Naives', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'naive_forecasts.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 6 figures saved.")
    return results


# ============================================================================
# STEP 7: ACF/PACF Analysis
# ============================================================================
def step7_acf_pacf(monthly):
    print("\n" + "="*70)
    print("STEP 7: ACF/PACF Analysis")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step7_acf_pacf')
    ensure_dir(save_dir)

    # Original series
    plot_acf_pacf(monthly, lags=50, title_prefix='Serie mensuelle originale',
                  save_path=os.path.join(save_dir, 'acf_pacf_original.png'))

    # Differenced series
    diff_monthly = monthly.diff().dropna()
    plot_acf_pacf(diff_monthly, lags=50, title_prefix='Serie mensuelle differenciee',
                  save_path=os.path.join(save_dir, 'acf_pacf_differenced.png'))

    print("Step 7 figures saved.")


# ============================================================================
# STEP 8: Statistical Models
# ============================================================================
def step8_statistical_models(train, test):
    print("\n" + "="*70)
    print("STEP 8: Statistical Models")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step8_statistical_models')
    ensure_dir(save_dir)

    n_test = len(test)
    results = []
    models_info = []

    # Model configurations: (name, order, seasonal_order)
    model_configs = [
        ('AR(2)', (2, 0, 0), None),
        ('MA(2)', (0, 0, 2), None),
        ('ARMA(2,2)', (2, 0, 2), None),
        ('ARIMA(2,1,2)', (2, 1, 2), None),
        ('SARIMA(1,1,1)(1,1,0,12)', (1, 1, 1), (1, 1, 0, 12)),
    ]

    forecasts = {}

    for name, order, seasonal_order in model_configs:
        try:
            print(f"\nFitting {name}...")
            if seasonal_order:
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = ARIMA(train, order=order,
                              enforce_stationarity=False, enforce_invertibility=False)

            fitted = model.fit()
            forecast = fitted.forecast(steps=n_test)
            forecasts[name] = forecast.values

            metrics = evaluate_forecast(test.values, forecast.values, name)
            metrics['AIC'] = round(fitted.aic, 2)
            metrics['BIC'] = round(fitted.bic, 2)
            results.append(metrics)
            models_info.append((name, fitted))
        except Exception as e:
            print(f"  Error with {name}: {e}")
            forecasts[name] = np.full(n_test, train.mean())
            results.append({'model': name, 'MAE': np.nan, 'RMSE': np.nan,
                           'MAPE': np.nan, 'AIC': np.nan, 'BIC': np.nan})

    # Plot all model forecasts
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()

    colors = ['#D32F2F', '#F57C00', '#388E3C', '#7B1FA2', '#00838F']
    for idx, (name, order, seasonal_order) in enumerate(model_configs):
        ax = axes[idx]
        ax.plot(train.index[-48:], train.values[-48:], label='Train', color='#1976D2', linewidth=1)
        ax.plot(test.index, test.values, label='Actual', color='#4CAF50', linewidth=1.5)
        if name in forecasts:
            ax.plot(test.index, forecasts[name][:n_test], label=name,
                    color=colors[idx], linewidth=1.5, linestyle='--')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Remove unused subplot
    axes[-1].set_visible(False)
    plt.suptitle('Modeles statistiques - Previsions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_models.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 8 figures saved.")
    return results


# ============================================================================
# STEP 9: Evaluation
# ============================================================================
def step9_evaluation(naive_results, stat_results):
    print("\n" + "="*70)
    print("STEP 9: Model Evaluation Comparison")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step9_evaluation')
    ensure_dir(save_dir)

    all_results = naive_results + stat_results
    df_results = pd.DataFrame(all_results)
    print("\n" + df_results.to_string(index=False))

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['MAE', 'RMSE', 'MAPE']
    colors_list = plt.cm.Set3(np.linspace(0, 1, len(df_results)))

    for idx, metric in enumerate(metrics):
        valid = df_results.dropna(subset=[metric])
        bars = axes[idx].barh(valid['model'], valid[metric], color=colors_list[:len(valid)])
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(metric)
        axes[idx].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Comparaison des modeles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 9 figures saved.")
    return df_results


# ============================================================================
# STEP 10: AutoARIMA
# ============================================================================
def step10_autoarima(train, test):
    print("\n" + "="*70)
    print("STEP 10: AutoARIMA")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step10_autoarima')
    ensure_dir(save_dir)

    # Fit auto_arima
    print("Fitting auto_arima (seasonal=True, m=12)...")
    auto_model = pm.auto_arima(train, seasonal=True, m=12,
                                suppress_warnings=True,
                                stepwise=True,
                                trace=False,
                                error_action='ignore')

    print(f"Selected order: {auto_model.order}")
    print(f"Seasonal order: {auto_model.seasonal_order}")
    print(f"AIC: {auto_model.aic():.2f}")

    # Forecast
    n_test = len(test)
    forecast, conf_int = auto_model.predict(n_periods=n_test, return_conf_int=True)

    metrics = evaluate_forecast(test.values, forecast, 'AutoARIMA')

    # Plot with confidence intervals
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(train.index[-48:], train.values[-48:], label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values, label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, forecast, label=f'AutoARIMA {auto_model.order}',
            color='#D32F2F', linewidth=1.5, linestyle='--')
    ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1],
                    alpha=0.2, color='#D32F2F', label='Intervalle de confiance 95%')
    ax.set_title(f'AutoARIMA - Ordre: {auto_model.order}, Saisonnier: {auto_model.seasonal_order}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix (USD/baril)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'autoarima_forecast.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 10 figures saved.")
    return metrics


# ============================================================================
# STEP 11: ML Models (XGBoost and Random Forest)
# ============================================================================
def step11_ml_models(df):
    print("\n" + "="*70)
    print("STEP 11: ML Models (XGBoost + Random Forest)")
    print("="*70)

    save_dir = os.path.join(FIGURES_DIR, 'step11_ml_models')
    ensure_dir(save_dir)

    # Create features
    df_feat = create_lag_features(df, target_col='Price',
                                  lags=[1, 7, 14, 30],
                                  rolling_windows=[7, 14, 30])

    # Drop NaN rows
    df_feat = df_feat.dropna()

    # Features and target
    feature_cols = [c for c in df_feat.columns if c != 'Price']
    X = df_feat[feature_cols]
    y = df_feat['Price']

    # 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")

    results = []

    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results.append(evaluate_forecast(y_test.values, xgb_pred, 'XGBoost'))

    # Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results.append(evaluate_forecast(y_test.values, rf_pred, 'Random Forest'))

    # Plot predictions
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # XGBoost predictions
    axes[0].plot(y_test.index, y_test.values, label='Actual', color='#4CAF50', linewidth=1.2)
    axes[0].plot(y_test.index, xgb_pred, label='XGBoost', color='#D32F2F', linewidth=1, alpha=0.8)
    axes[0].set_title('XGBoost - Predictions', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Prix (USD)')

    # Random Forest predictions
    axes[1].plot(y_test.index, y_test.values, label='Actual', color='#4CAF50', linewidth=1.2)
    axes[1].plot(y_test.index, rf_pred, label='Random Forest', color='#7B1FA2', linewidth=1, alpha=0.8)
    axes[1].set_title('Random Forest - Predictions', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Prix (USD)')
    axes[1].set_xlabel('Date')

    plt.suptitle('Modeles ML - Previsions one-step-ahead', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ml_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # XGBoost feature importance
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    xgb_top = xgb_importance.nlargest(15)
    xgb_top.plot(kind='barh', ax=axes[0], color='#EF5350')
    axes[0].set_title('XGBoost - Importance des features (Top 15)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Importance')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Random Forest feature importance
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
    rf_top = rf_importance.nlargest(15)
    rf_top.plot(kind='barh', ax=axes[1], color='#AB47BC')
    axes[1].set_title('Random Forest - Importance des features (Top 15)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("Step 11 figures saved.")
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*70)
    print(" BRENT OIL PRICE - TIME SERIES ANALYSIS AND FORECASTING")
    print("="*70)

    # Step 1
    df = step1_visualization()

    # Step 2
    monthly = step2_decomposition(df)

    # Step 3
    step3_stationarity(df)

    # Step 4
    log_series = step4_transformations(df)

    # Step 5
    train, test = step5_train_test_split(df, monthly)

    # Step 6
    naive_results = step6_naive_forecasting(train, test)

    # Step 7
    step7_acf_pacf(monthly)

    # Step 8
    stat_results = step8_statistical_models(train, test)

    # Step 9
    step9_evaluation(naive_results, stat_results)

    # Step 10
    step10_autoarima(train, test)

    # Step 11
    ml_results = step11_ml_models(df)

    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE - All figures saved to figures/ directory")
    print("="*70)


if __name__ == '__main__':
    main()
