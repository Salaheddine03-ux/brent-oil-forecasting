"""
Brent Oil Price Time Series Analysis - V2 (Corrected)
======================================================
Fixes three critical issues from the original analysis:

1. UNFAIR COMPARISON: ML models used one-step-ahead prediction while statistical
   models used multi-step forecasting. Fix: Implement recursive multi-step
   forecasting for ML models.

2. LAG_1 DOMINANCE: lag_1 feature made ML models essentially copy yesterday's
   price, hiding real pattern learning. Fix: Remove lag_1, use lags starting
   from 7, and add log-return features.

3. DECOMPOSITION CHOICE: Additive decomposition was hardcoded without
   justification. Fix: Auto-detect whether additive or multiplicative is
   appropriate based on coefficient of variation analysis.
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pmdarima as pm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data, evaluate_forecast

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
# FIX 3: AUTOMATIC DECOMPOSITION SELECTION
# ============================================================================
def step2_auto_decomposition(df):
    """
    Automatically select additive vs multiplicative decomposition.

    Method:
    - Resample to monthly data
    - Split into equal segments (4 segments)
    - Compute std of residuals in each segment after removing trend
    - If std grows proportionally with the level (correlation > 0.5),
      use multiplicative; otherwise use additive
    """
    print("\n" + "=" * 70)
    print("STEP 2 (V2): Automatic Decomposition Selection")
    print("=" * 70)

    save_dir = os.path.join(FIGURES_DIR, 'step2_decomposition')
    ensure_dir(save_dir)

    # Resample to monthly
    monthly = df['Price'].resample('MS').mean()

    # Remove trend using rolling mean
    trend = monthly.rolling(window=12, center=True).mean()

    # Get detrended series (residuals from trend)
    detrended = monthly - trend
    detrended = detrended.dropna()

    # Get corresponding level (trend values at same indices)
    trend_at_detrended = trend.loc[detrended.index]

    # Split into 4 equal segments
    n_segments = 4
    segment_size = len(detrended) // n_segments

    segment_stds = []
    segment_levels = []

    print(f"\nAnalysis of {n_segments} segments:")
    print("-" * 50)

    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < n_segments - 1 else len(detrended)

        segment_residuals = detrended.iloc[start_idx:end_idx]
        segment_trend = trend_at_detrended.iloc[start_idx:end_idx]

        std_val = segment_residuals.std()
        level_val = segment_trend.mean()

        segment_stds.append(std_val)
        segment_levels.append(level_val)

        print(f"  Segment {i+1}: Level={level_val:.2f}, "
              f"Std of detrended={std_val:.4f}")

    # Compute correlation between level and std
    correlation = np.corrcoef(segment_levels, segment_stds)[0, 1]

    # Decision rule: if correlation > 0.5, seasonal amplitude grows with level
    if correlation > 0.5:
        decomp_type = 'multiplicative'
        reasoning = (
            f"Correlation between segment level and residual std = {correlation:.3f} > 0.5.\n"
            f"The seasonal amplitude grows proportionally with the price level.\n"
            f"This indicates multiplicative seasonality."
        )
    else:
        decomp_type = 'additive'
        reasoning = (
            f"Correlation between segment level and residual std = {correlation:.3f} <= 0.5.\n"
            f"The seasonal amplitude does NOT grow proportionally with the price level.\n"
            f"This indicates additive seasonality is appropriate."
        )

    print(f"\n{'=' * 50}")
    print(f"DECISION: {decomp_type.upper()} decomposition")
    print(f"{'=' * 50}")
    print(reasoning)

    # Apply the selected decomposition
    decomposition = seasonal_decompose(monthly, model=decomp_type, period=12)

    # Plot decomposition with decision annotation
    fig, axes = plt.subplots(5, 1, figsize=(14, 16))

    # First subplot: decision explanation
    axes[0].axis('off')
    decision_text = (
        f"Automatic Decomposition Type Selection\n"
        f"{'=' * 50}\n\n"
        f"Method: Split series into {n_segments} segments, compute std of "
        f"detrended residuals in each.\n"
        f"If std grows with the price level (corr > 0.5) -> multiplicative\n"
        f"If std stays constant (corr <= 0.5) -> additive\n\n"
        f"Results:\n"
    )
    for i in range(n_segments):
        decision_text += (
            f"  Segment {i+1}: Level = {segment_levels[i]:.1f}, "
            f"Std = {segment_stds[i]:.3f}\n"
        )
    decision_text += (
        f"\nCorrelation(level, std) = {correlation:.3f}\n"
        f"Decision: {decomp_type.upper()}"
    )
    axes[0].text(0.05, 0.95, decision_text, transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    axes[1].plot(decomposition.observed, color='#1976D2')
    axes[1].set_title('Observe', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Prix')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(decomposition.trend, color='#F57C00')
    axes[2].set_title('Tendance', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Prix')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(decomposition.seasonal, color='#388E3C')
    axes[3].set_title('Saisonnalite', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Prix')
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(decomposition.resid, color='#D32F2F')
    axes[4].set_title('Residu', fontsize=12, fontweight='bold')
    axes[4].set_ylabel('Prix')
    axes[4].grid(True, alpha=0.3)

    plt.suptitle(
        f'Decomposition de la serie temporelle '
        f'({decomp_type.capitalize()}, selection automatique)',
        fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'decomposition_auto_selection.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\nStep 2 (V2) figure saved: decomposition_auto_selection.png")
    return monthly, decomp_type


# ============================================================================
# FIX 1 & 2: FAIR ML FORECASTING
# ============================================================================
def build_features_from_history(prices, lags, rolling_windows):
    """
    Build feature vector from a price history array.

    Features:
    - Lag features (lag_7, lag_14, lag_30) - NO lag_1
    - Rolling mean and std (windows 7, 14, 30)
    - Log-return features (log(price_t) - log(price_t-k))
    - Calendar features from the last date are NOT used here
      since we work with monthly data (month is implicit)

    Args:
        prices: list or array of historical prices (most recent at end)
        lags: list of lag periods to use
        rolling_windows: list of rolling window sizes

    Returns:
        dict of feature_name: value
    """
    features = {}
    n = len(prices)

    # Lag features (lag_7 means price 7 steps back from current)
    for lag in lags:
        if n > lag:
            features[f'lag_{lag}'] = prices[-(lag + 1)]
        else:
            features[f'lag_{lag}'] = prices[0]

    # Rolling statistics (computed on the last `window` prices, shifted by 1)
    for window in rolling_windows:
        if n > window:
            window_data = prices[-(window + 1):-1]
        else:
            window_data = prices[:-1] if len(prices) > 1 else prices
        features[f'rolling_mean_{window}'] = np.mean(window_data)
        features[f'rolling_std_{window}'] = np.std(window_data) if len(window_data) > 1 else 0.0

    # Log-return features
    current_price = prices[-1]
    for lag in lags:
        if n > lag and prices[-(lag + 1)] > 0 and current_price > 0:
            features[f'log_return_{lag}'] = (
                np.log(current_price) - np.log(prices[-(lag + 1)])
            )
        else:
            features[f'log_return_{lag}'] = 0.0

    return features


def create_ml_features_df(series, lags, rolling_windows):
    """
    Create a DataFrame with ML features from a price series.
    No lag_1 -- uses lags starting from 7.
    Adds log-return features.

    Args:
        series: pandas Series with DatetimeIndex
        lags: list of lag values (e.g., [7, 14, 30])
        rolling_windows: list of rolling window sizes

    Returns:
        DataFrame with features and target
    """
    df_feat = pd.DataFrame(index=series.index)
    df_feat['Price'] = series.values

    # Lag features (no lag_1!)
    for lag in lags:
        df_feat[f'lag_{lag}'] = series.shift(lag).values

    # Rolling statistics (shifted by 1 to avoid data leakage)
    for window in rolling_windows:
        df_feat[f'rolling_mean_{window}'] = (
            series.shift(1).rolling(window=window).mean().values
        )
        df_feat[f'rolling_std_{window}'] = (
            series.shift(1).rolling(window=window).std().values
        )

    # Log-return features
    for lag in lags:
        df_feat[f'log_return_{lag}'] = (
            np.log(series) - np.log(series.shift(lag))
        ).values

    return df_feat


def recursive_forecast(model, history_prices, n_steps, feature_names, lags, rolling_windows):
    """
    Perform recursive multi-step forecasting.

    The model predicts t+1, that prediction becomes part of history for t+2,
    and so on for the entire test horizon.

    Args:
        model: trained sklearn-compatible model
        history_prices: list of training prices (will be extended with predictions)
        n_steps: number of steps to forecast
        feature_names: ordered list of feature names the model expects
        lags: lag values used in features
        rolling_windows: rolling window sizes used in features

    Returns:
        list of predictions
    """
    predictions = []
    current_prices = list(history_prices)  # copy of training prices

    for step in range(n_steps):
        # Build features from current price history
        features_dict = build_features_from_history(
            current_prices, lags, rolling_windows
        )

        # Create feature vector in the correct order
        feature_vector = [features_dict[name] for name in feature_names]

        # Predict next value
        pred = model.predict([feature_vector])[0]
        predictions.append(pred)

        # Use prediction as next "observed" price for recursive forecasting
        current_prices.append(pred)

    return predictions


def step11_fair_ml_models(df, train_monthly, test_monthly):
    """
    Step 11 (V2): Fair ML forecasting with recursive multi-step prediction.

    Fixes:
    - Removes lag_1 (uses lags 7, 14, 30 only)
    - Adds log-return features
    - Uses recursive multi-step forecasting (not one-step-ahead)
    - Trains on monthly data to match statistical models
    """
    print("\n" + "=" * 70)
    print("STEP 11 (V2): Fair ML Models - Recursive Multi-Step Forecasting")
    print("=" * 70)

    save_dir = os.path.join(FIGURES_DIR, 'step11_ml_models')
    ensure_dir(save_dir)

    # Configuration - NO lag_1
    lags = [7, 14, 30]
    rolling_windows = [7, 14, 30]

    print(f"\nFeature configuration:")
    print(f"  Lags: {lags} (lag_1 REMOVED to prevent persistence copying)")
    print(f"  Rolling windows: {rolling_windows}")
    print(f"  Log-return features: log_return_7, log_return_14, log_return_30")

    # Create features on monthly data (to match statistical models)
    monthly_full = df['Price'].resample('MS').mean()

    df_feat = create_ml_features_df(monthly_full, lags, rolling_windows)
    df_feat = df_feat.dropna()

    # Align with train/test split
    feature_cols = [c for c in df_feat.columns if c != 'Price']
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")

    # Use same split as statistical models
    train_end = train_monthly.index[-1]
    test_start = test_monthly.index[0]

    df_train = df_feat.loc[:train_end]
    df_test = df_feat.loc[test_start:]

    X_train = df_train[feature_cols]
    y_train = df_train['Price']

    n_test = len(test_monthly)

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Test horizon (multi-step): {n_test} months")

    # Get training prices for recursive forecasting
    training_prices = list(monthly_full.loc[:train_end].values)

    results = []

    # --- XGBoost ---
    print("\n  Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    # Recursive multi-step forecast
    print("  Performing recursive multi-step forecast (XGBoost)...")
    xgb_recursive = recursive_forecast(
        xgb_model, training_prices, n_test, feature_cols, lags, rolling_windows
    )
    xgb_metrics = evaluate_forecast(
        test_monthly.values, np.array(xgb_recursive),
        'XGBoost (Recursive Multi-Step)'
    )
    results.append(xgb_metrics)

    # --- Random Forest ---
    print("\n  Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Recursive multi-step forecast
    print("  Performing recursive multi-step forecast (Random Forest)...")
    rf_recursive = recursive_forecast(
        rf_model, training_prices, n_test, feature_cols, lags, rolling_windows
    )
    rf_metrics = evaluate_forecast(
        test_monthly.values, np.array(rf_recursive),
        'Random Forest (Recursive Multi-Step)'
    )
    results.append(rf_metrics)

    # --- Plot ML recursive forecasts ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(train_monthly.index[-48:], train_monthly.values[-48:],
                 label='Train', color='#1976D2', linewidth=1.2)
    axes[0].plot(test_monthly.index, test_monthly.values,
                 label='Actual', color='#4CAF50', linewidth=1.5)
    axes[0].plot(test_monthly.index[:n_test], xgb_recursive,
                 label='XGBoost (Recursive)', color='#D32F2F',
                 linewidth=1.5, linestyle='--')
    axes[0].set_title('XGBoost - Recursive Multi-Step Forecast', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Prix (USD)')

    axes[1].plot(train_monthly.index[-48:], train_monthly.values[-48:],
                 label='Train', color='#1976D2', linewidth=1.2)
    axes[1].plot(test_monthly.index, test_monthly.values,
                 label='Actual', color='#4CAF50', linewidth=1.5)
    axes[1].plot(test_monthly.index[:n_test], rf_recursive,
                 label='Random Forest (Recursive)', color='#7B1FA2',
                 linewidth=1.5, linestyle='--')
    axes[1].set_title('Random Forest - Recursive Multi-Step Forecast', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Prix (USD)')
    axes[1].set_xlabel('Date')

    plt.suptitle(
        'Modeles ML - Previsions multi-step recursives\n'
        '(lag_1 supprime, log-returns ajoutes, comparaison equitable)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ml_recursive_forecast.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Feature importance ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    xgb_importance.sort_values().plot(kind='barh', ax=axes[0], color='#EF5350')
    axes[0].set_title('XGBoost - Feature Importance\n(No lag_1)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Importance')
    axes[0].grid(True, alpha=0.3, axis='x')

    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
    rf_importance.sort_values().plot(kind='barh', ax=axes[1], color='#AB47BC')
    axes[1].set_title('Random Forest - Feature Importance\n(No lag_1)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_v2.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\nStep 11 (V2) figures saved: ml_recursive_forecast.png, feature_importance_v2.png")
    return results, xgb_recursive, rf_recursive


# ============================================================================
# FAIR COMPARISON: All models on same test set with multi-step forecasting
# ============================================================================
def fair_comparison(train_monthly, test_monthly, ml_results, xgb_forecast, rf_forecast):
    """
    Compare ALL models fairly using multi-step forecasting on the same test set.
    Statistical models (ARIMA, SARIMA, AutoARIMA) naturally do multi-step.
    ML models now also do multi-step via recursive forecasting.
    """
    print("\n" + "=" * 70)
    print("FAIR MODEL COMPARISON (All Multi-Step)")
    print("=" * 70)

    save_dir_ml = os.path.join(FIGURES_DIR, 'step11_ml_models')
    save_dir_eval = os.path.join(FIGURES_DIR, 'step9_evaluation')
    ensure_dir(save_dir_ml)
    ensure_dir(save_dir_eval)

    n_test = len(test_monthly)
    all_results = []
    all_forecasts = {}

    # --- Statistical models (multi-step by nature) ---
    stat_models = [
        ('ARIMA(2,1,2)', (2, 1, 2), None),
        ('SARIMA(1,1,1)(1,1,0,12)', (1, 1, 1), (1, 1, 0, 12)),
    ]

    for name, order, seasonal_order in stat_models:
        try:
            print(f"\nFitting {name}...")
            if seasonal_order:
                model = SARIMAX(train_monthly, order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            else:
                model = ARIMA(train_monthly, order=order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_test)
            all_forecasts[name] = forecast.values
            metrics = evaluate_forecast(test_monthly.values, forecast.values, name)
            all_results.append(metrics)
        except Exception as e:
            print(f"  Error with {name}: {e}")
            all_forecasts[name] = np.full(n_test, train_monthly.mean())
            all_results.append({'model': name, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan})

    # AutoARIMA
    try:
        print("\nFitting AutoARIMA...")
        auto_model = pm.auto_arima(train_monthly, seasonal=True, m=12,
                                    suppress_warnings=True, stepwise=True,
                                    trace=False, error_action='ignore')
        auto_forecast = auto_model.predict(n_periods=n_test)
        all_forecasts['AutoARIMA'] = auto_forecast
        metrics = evaluate_forecast(test_monthly.values, auto_forecast, 'AutoARIMA')
        all_results.append(metrics)
        print(f"  AutoARIMA order: {auto_model.order}, seasonal: {auto_model.seasonal_order}")
    except Exception as e:
        print(f"  Error with AutoARIMA: {e}")
        all_forecasts['AutoARIMA'] = np.full(n_test, train_monthly.mean())
        all_results.append({'model': 'AutoARIMA', 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan})

    # ML models (recursive multi-step -- already computed)
    all_forecasts['XGBoost (Recursive)'] = xgb_forecast
    all_forecasts['Random Forest (Recursive)'] = rf_forecast
    all_results.extend(ml_results)

    # --- Plot: ML recursive vs ARIMA vs SARIMA ---
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(train_monthly.index[-36:], train_monthly.values[-36:],
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test_monthly.index, test_monthly.values,
            label='Actual', color='#4CAF50', linewidth=2)

    colors = {'ARIMA(2,1,2)': '#F57C00',
              'SARIMA(1,1,1)(1,1,0,12)': '#00838F',
              'AutoARIMA': '#795548',
              'XGBoost (Recursive)': '#D32F2F',
              'Random Forest (Recursive)': '#7B1FA2'}

    for name, forecast in all_forecasts.items():
        ax.plot(test_monthly.index[:len(forecast)], forecast[:n_test],
                label=name, color=colors.get(name, '#333'),
                linewidth=1.5, linestyle='--')

    ax.set_title(
        'Comparaison equitable: Tous les modeles en multi-step\n'
        '(ML: recursif, Statistiques: natif)',
        fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix (USD/baril)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_ml, 'fair_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Final comparison table figure ---
    df_results = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("FINAL FAIR COMPARISON TABLE (All Multi-Step Forecasting)")
    print("=" * 70)
    print(df_results.to_string(index=False))

    # Create table as figure
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Create table
    table_data = []
    headers = ['Model', 'MAE', 'RMSE', 'MAPE (%)']
    for _, row in df_results.iterrows():
        table_data.append([
            row['model'],
            f"{row['MAE']:.2f}" if not np.isnan(row['MAE']) else 'N/A',
            f"{row['RMSE']:.2f}" if not np.isnan(row['RMSE']) else 'N/A',
            f"{row['MAPE']:.2f}" if not np.isnan(row['MAPE']) else 'N/A',
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.4, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1976D2')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(table_data)):
        color = '#E3F2FD' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title(
        'Comparaison equitable des modeles\n'
        '(Tous en prevision multi-step sur la meme periode de test)',
        fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_eval, 'fair_model_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Bar chart comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE']

    for idx, metric in enumerate(metrics_to_plot):
        valid = df_results.dropna(subset=[metric]).sort_values(metric)
        bar_colors = ['#4CAF50' if 'Recursive' in m else '#1976D2'
                      for m in valid['model']]
        bars = axes[idx].barh(valid['model'], valid[metric], color=bar_colors)
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(metric)
        axes[idx].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar_item in bars:
            width = bar_item.get_width()
            axes[idx].text(width, bar_item.get_y() + bar_item.get_height() / 2,
                           f' {width:.2f}', va='center', fontsize=9)

    plt.suptitle(
        'Comparaison equitable - Metriques\n'
        '(Vert = ML recursif, Bleu = Statistique)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_eval, 'fair_model_comparison_bars.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("\nFair comparison figures saved:")
    print("  - figures/step11_ml_models/fair_comparison.png")
    print("  - figures/step9_evaluation/fair_model_comparison.png")
    print("  - figures/step9_evaluation/fair_model_comparison_bars.png")

    return df_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("=" * 70)
    print(" BRENT OIL PRICE - CORRECTED ANALYSIS (V2)")
    print("=" * 70)
    print("\nFixes applied:")
    print("  1. Recursive multi-step forecasting for ML models (fair comparison)")
    print("  2. lag_1 removed; log-return features added (forces real learning)")
    print("  3. Automatic additive/multiplicative decomposition selection")
    print()

    # Load data
    df = load_data(os.path.join(DATA_DIR, 'BrentOilPrices.csv'))
    print(f"Dataset loaded: {len(df)} observations")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Fix 3: Auto decomposition selection
    monthly, decomp_type = step2_auto_decomposition(df)

    # Train/test split (same as original: 80/20 on monthly)
    n = len(monthly)
    split_idx = int(n * 0.8)
    train_monthly = monthly[:split_idx]
    test_monthly = monthly[split_idx:]
    print(f"\nTrain/test split: {len(train_monthly)}/{len(test_monthly)} months")

    # Fix 1 & 2: Fair ML models
    ml_results, xgb_forecast, rf_forecast = step11_fair_ml_models(
        df, train_monthly, test_monthly
    )

    # Fair comparison of all models
    comparison = fair_comparison(
        train_monthly, test_monthly, ml_results, xgb_forecast, rf_forecast
    )

    print("\n" + "=" * 70)
    print(" V2 ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey differences from V1:")
    print(f"  - Decomposition type: {decomp_type} (auto-selected, was hardcoded additive)")
    print("  - ML models use recursive multi-step (was one-step-ahead)")
    print("  - lag_1 removed, log-returns added (was lag_1 dominant)")
    print("  - All models now comparable on same multi-step test horizon")
    print("\nNew figures generated:")
    print("  - figures/step2_decomposition/decomposition_auto_selection.png")
    print("  - figures/step11_ml_models/ml_recursive_forecast.png")
    print("  - figures/step11_ml_models/feature_importance_v2.png")
    print("  - figures/step11_ml_models/fair_comparison.png")
    print("  - figures/step9_evaluation/fair_model_comparison.png")
    print("  - figures/step9_evaluation/fair_model_comparison_bars.png")


if __name__ == '__main__':
    main()
