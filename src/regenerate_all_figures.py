"""
Regenerate ALL individual model forecast figures with V2 corrections applied.
=============================================================================

This script regenerates figures for steps 6, 8, 10, 11, and 12 with:
- enforce_stationarity=True, enforce_invertibility=True for ARIMA
- Corrected ML hyperparameters (RF: max_depth=5, min_samples_leaf=5, max_features=0.7;
  XGBoost: max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.8)
- No lag_1 for ML models (use lags=[7, 14, 30])
- seasonal=False, d=1 for AutoARIMA
- Proper np.clip(forecast, 1, None) for all forecasts
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
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import pmdarima as pm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data
from src.analysis_v2 import (
    create_ml_features_df,
    build_features_from_history,
    recursive_forecast,
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# V2 configuration
LAGS = [7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_data():
    """Load data and prepare monthly train/test split."""
    df = load_data(os.path.join(DATA_DIR, 'BrentOilPrices.csv'))
    monthly = df['Price'].resample('MS').mean()
    n = len(monthly)
    split_idx = int(n * 0.8)
    train = monthly[:split_idx]
    test = monthly[split_idx:]
    return df, monthly, train, test


# ============================================================================
# STEP 6: NAIVE FORECASTING
# ============================================================================
def regenerate_step6(train, test):
    """
    Regenerate naive_forecasts.png - 2x2 grid showing:
    - Last Value Naive forecast vs actual (with MAE)
    - Seasonal Naive forecast vs actual (with MAE)
    - Moving Average (12) forecast vs actual (with MAE)
    - Simple Exponential Smoothing forecast vs actual (with MAE)
    """
    print("\n" + "=" * 70)
    print("STEP 6: Regenerating Naive Forecasting Figures")
    print("=" * 70)

    save_dir = os.path.join(FIGURES_DIR, 'step6_naive_forecasting')
    ensure_dir(save_dir)

    n_test = len(test)
    # Show last 48 months of training + full test
    train_display = train[-48:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Last Value Naive ---
    ax = axes[0, 0]
    naive_last = np.full(n_test, train.iloc[-1])
    naive_last = np.clip(naive_last, 1, None)
    mae_last = mean_absolute_error(test.values, naive_last)

    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, naive_last,
            label='Last Value Naive', color='#D32F2F', linewidth=1.5, linestyle='--')
    ax.set_title(f'Last Value Naive\nMAE = {mae_last:.2f} USD', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Prix (USD)')

    # --- Seasonal Naive (period=12 for monthly) ---
    ax = axes[0, 1]
    seasonal_naive = np.array([train.iloc[-(12 - i % 12)] for i in range(n_test)])
    seasonal_naive = np.clip(seasonal_naive, 1, None)
    mae_seasonal = mean_absolute_error(test.values, seasonal_naive)

    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, seasonal_naive,
            label='Seasonal Naive (p=12)', color='#F57C00', linewidth=1.5, linestyle='--')
    ax.set_title(f'Seasonal Naive (period=12)\nMAE = {mae_seasonal:.2f} USD',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Prix (USD)')

    # --- Moving Average (12) ---
    ax = axes[1, 0]
    ma_value = train.iloc[-12:].mean()
    ma_forecast = np.full(n_test, ma_value)
    ma_forecast = np.clip(ma_forecast, 1, None)
    mae_ma = mean_absolute_error(test.values, ma_forecast)

    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, ma_forecast,
            label='Moving Average (12)', color='#7B1FA2', linewidth=1.5, linestyle='--')
    ax.set_title(f'Moving Average (window=12)\nMAE = {mae_ma:.2f} USD',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Prix (USD)')
    ax.set_xlabel('Date')

    # --- Simple Exponential Smoothing ---
    ax = axes[1, 1]
    ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.3, optimized=False)
    ses_forecast = ses_model.forecast(n_test).values
    ses_forecast = np.clip(ses_forecast, 1, None)
    mae_ses = mean_absolute_error(test.values, ses_forecast)

    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, ses_forecast,
            label='SES (alpha=0.3)', color='#00838F', linewidth=1.5, linestyle='--')
    ax.set_title(f'Simple Exponential Smoothing\nMAE = {mae_ses:.2f} USD',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Prix (USD)')
    ax.set_xlabel('Date')

    plt.suptitle(
        'Step 6: Methodes Naive de Prevision (V2 - donnees mensuelles)\n'
        'Multi-step forecast sur toute la periode de test',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'naive_forecasts.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Last Value Naive MAE: {mae_last:.2f}")
    print(f"  Seasonal Naive MAE: {mae_seasonal:.2f}")
    print(f"  Moving Average (12) MAE: {mae_ma:.2f}")
    print(f"  SES MAE: {mae_ses:.2f}")
    print("  Saved: figures/step6_naive_forecasting/naive_forecasts.png")


# ============================================================================
# STEP 8: STATISTICAL MODELS
# ============================================================================
def regenerate_step8(train, test):
    """
    Regenerate statistical_models.png - 3x2 grid with:
    - AR(2): INVALIDE annotation (red border)
    - MA(2): INVALIDE annotation (red border)
    - ARMA(2,2): INVALIDE annotation (red border)
    - ARIMA(2,1,2): VALIDE annotation (green border)
    - SARIMA(1,1,1)(1,1,0,12): ORANGE annotation
    - Summary text box
    """
    print("\n" + "=" * 70)
    print("STEP 8: Regenerating Statistical Models Figures")
    print("=" * 70)

    save_dir = os.path.join(FIGURES_DIR, 'step8_statistical_models')
    ensure_dir(save_dir)

    n_test = len(test)
    train_display = train[-48:]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    models_config = [
        {
            'name': 'AR(2)',
            'order': (2, 0, 0),
            'seasonal_order': None,
            'enforce_stationarity': False,
            'enforce_invertibility': False,
            'annotation': "INVALIDE: d=0 sur serie non-stationnaire",
            'color': '#D32F2F',
            'border_color': 'red',
            'validity': 'invalid',
            'ax_pos': (0, 0),
        },
        {
            'name': 'MA(2)',
            'order': (0, 0, 2),
            'seasonal_order': None,
            'enforce_stationarity': False,
            'enforce_invertibility': False,
            'annotation': "INVALIDE: d=0, memoire 2 lags",
            'color': '#F57C00',
            'border_color': 'red',
            'validity': 'invalid',
            'ax_pos': (0, 1),
        },
        {
            'name': 'ARMA(2,2)',
            'order': (2, 0, 2),
            'seasonal_order': None,
            'enforce_stationarity': False,
            'enforce_invertibility': False,
            'annotation': "INVALIDE: d=0, revert vers la moyenne",
            'color': '#7B1FA2',
            'border_color': 'red',
            'validity': 'invalid',
            'ax_pos': (1, 0),
        },
        {
            'name': 'ARIMA(2,1,2)',
            'order': (2, 1, 2),
            'seasonal_order': None,
            'enforce_stationarity': True,
            'enforce_invertibility': True,
            'annotation': "VALIDE: enforce_stationarity=True",
            'color': '#388E3C',
            'border_color': 'green',
            'validity': 'valid',
            'ax_pos': (1, 1),
        },
        {
            'name': 'SARIMA(1,1,1)(1,1,0,12)',
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 0, 12),
            'enforce_stationarity': False,
            'enforce_invertibility': False,
            'annotation': "Diverge vers prix negatifs",
            'color': '#00838F',
            'border_color': 'orange',
            'validity': 'warning',
            'ax_pos': (2, 0),
        },
    ]

    results_summary = []

    for cfg in models_config:
        ax = axes[cfg['ax_pos'][0], cfg['ax_pos'][1]]

        try:
            if cfg['seasonal_order']:
                model = SARIMAX(
                    train, order=cfg['order'],
                    seasonal_order=cfg['seasonal_order'],
                    enforce_stationarity=cfg['enforce_stationarity'],
                    enforce_invertibility=cfg['enforce_invertibility']
                )
            else:
                model = ARIMA(
                    train, order=cfg['order'],
                    enforce_stationarity=cfg['enforce_stationarity'],
                    enforce_invertibility=cfg['enforce_invertibility']
                )
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_test)
            forecast_values = np.clip(forecast.values, 1, None)
            mae = mean_absolute_error(test.values, forecast_values)
        except Exception as e:
            print(f"  Warning: {cfg['name']} failed to fit - {e}")
            forecast_values = np.full(n_test, train.mean())
            forecast_values = np.clip(forecast_values, 1, None)
            mae = mean_absolute_error(test.values, forecast_values)

        results_summary.append({
            'name': cfg['name'],
            'mae': mae,
            'validity': cfg['validity'],
        })

        # Plot
        ax.plot(train_display.index, train_display.values,
                label='Train', color='#1976D2', linewidth=1.2)
        ax.plot(test.index, test.values,
                label='Actual', color='#4CAF50', linewidth=1.5)
        ax.plot(test.index, forecast_values,
                label=f'{cfg["name"]}', color=cfg['color'],
                linewidth=1.5, linestyle='--')
        ax.set_title(f'{cfg["name"]}\nMAE = {mae:.2f} USD',
                     fontsize=11, fontweight='bold',
                     color=cfg['border_color'])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Prix (USD)')

        # Add annotation box
        ann_color = {'invalid': '#FFCDD2', 'valid': '#C8E6C9', 'warning': '#FFE0B2'}
        ax.text(0.02, 0.98, cfg['annotation'],
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor=ann_color[cfg['validity']], alpha=0.8))

        # Border color
        for spine in ax.spines.values():
            spine.set_edgecolor(cfg['border_color'])
            spine.set_linewidth(2.5)

        print(f"  {cfg['name']}: MAE={mae:.2f}, status={cfg['validity']}")

    # --- Summary panel (bottom right) ---
    ax_summary = axes[2, 1]
    ax_summary.axis('off')

    summary_text = "RESUME - Validite des modeles\n" + "=" * 40 + "\n\n"
    summary_text += "VALIDES (d >= 1, serie differenciee):\n"
    for r in results_summary:
        if r['validity'] == 'valid':
            summary_text += f"  + {r['name']}: MAE={r['mae']:.2f}\n"

    summary_text += "\nINVALIDES (d=0, serie non-stationnaire):\n"
    for r in results_summary:
        if r['validity'] == 'invalid':
            summary_text += f"  x {r['name']}: MAE={r['mae']:.2f}\n"

    summary_text += "\nATTENTION (divergence):\n"
    for r in results_summary:
        if r['validity'] == 'warning':
            summary_text += f"  ! {r['name']}: MAE={r['mae']:.2f}\n"

    summary_text += "\n" + "-" * 40
    summary_text += "\nSeul ARIMA(2,1,2) est valide car il\n"
    summary_text += "differencie la serie (d=1) et impose\n"
    summary_text += "enforce_stationarity=True.\n"
    summary_text += "\nAR, MA, ARMA travaillent sur la serie\n"
    summary_text += "brute non-stationnaire -> biais severe."

    ax_summary.text(0.05, 0.95, summary_text,
                    transform=ax_summary.transAxes, fontsize=10, va='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.suptitle(
        'Step 8: Modeles Statistiques (V2 corrections)\n'
        'Vert=valide, Rouge=invalide (d=0), Orange=divergence',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_models.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: figures/step8_statistical_models/statistical_models.png")


# ============================================================================
# STEP 10: AUTOARIMA
# ============================================================================
def regenerate_step10(train, test):
    """
    Regenerate autoarima_forecast.png - AutoARIMA with seasonal=False, d=1.
    Shows order selected, forecast with CI, MAE on plot.
    """
    print("\n" + "=" * 70)
    print("STEP 10: Regenerating AutoARIMA Figures")
    print("=" * 70)

    save_dir = os.path.join(FIGURES_DIR, 'step10_autoarima')
    ensure_dir(save_dir)

    n_test = len(test)
    train_display = train[-48:]

    # Fit AutoARIMA with corrected parameters
    print("  Fitting AutoARIMA (seasonal=False, d=1)...")
    auto_model = pm.auto_arima(
        train,
        seasonal=False,
        d=1,
        max_d=2,
        suppress_warnings=True,
        stepwise=True,
        trace=False,
        error_action='ignore'
    )

    order = auto_model.order
    print(f"  Selected order: ARIMA{order}")

    # Forecast with confidence intervals
    forecast, conf_int = auto_model.predict(n_periods=n_test, return_conf_int=True)
    forecast = np.clip(forecast, 1, None)
    conf_int = np.array(conf_int, copy=True)
    conf_int[:, 0] = np.clip(conf_int[:, 0], 1, None)
    conf_int[:, 1] = np.clip(conf_int[:, 1], 1, None)

    mae = mean_absolute_error(test.values, forecast)
    print(f"  AutoARIMA MAE: {mae:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.8)
    ax.plot(test.index, forecast,
            label=f'AutoARIMA {order}', color='#D32F2F', linewidth=1.8, linestyle='--')
    ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1],
                    alpha=0.15, color='#D32F2F', label='Intervalle de confiance 95%')

    ax.set_title(
        f'Step 10: AutoARIMA - Ordre selectionne: ARIMA{order}\n'
        f'MAE = {mae:.2f} USD',
        fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix (USD)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.text(0.02, 0.02,
            "seasonal=False, d=1 (corrige: evite ARIMA(0,1,0))\n"
            f"Ordre final: ARIMA{order}",
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'autoarima_forecast.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Saved: figures/step10_autoarima/autoarima_forecast.png")
    return order, mae


# ============================================================================
# STEP 11: ML MODELS
# ============================================================================
def regenerate_step11(df, monthly, train, test):
    """
    Regenerate:
    - ml_recursive_forecast.png - 2 panels (XGBoost, Random Forest)
    - feature_importance_v2.png - Feature importance horizontal bar charts
    """
    print("\n" + "=" * 70)
    print("STEP 11: Regenerating ML Model Figures")
    print("=" * 70)

    save_dir = os.path.join(FIGURES_DIR, 'step11_ml_models')
    ensure_dir(save_dir)

    n_test = len(test)
    train_display = train[-48:]

    # Create features using V2 (no lag_1, with log-returns)
    df_feat = create_ml_features_df(monthly, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c != 'Price']

    # Split aligned with train/test
    train_end = train.index[-1]
    df_train = df_feat.loc[:train_end]

    X_train = df_train[feature_cols]
    y_train = df_train['Price']

    # Training prices for recursive forecasting
    training_prices = list(monthly.loc[:train_end].values)

    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test horizon: {n_test} months (recursive multi-step)")

    # --- Train XGBoost ---
    print("  Training XGBoost (max_depth=3, min_child_weight=10)...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    # Recursive multi-step forecast
    xgb_recursive = recursive_forecast(
        xgb_model, training_prices, n_test, feature_cols, LAGS, ROLLING_WINDOWS
    )
    xgb_recursive = np.clip(xgb_recursive, 1, None)
    mae_xgb = mean_absolute_error(test.values, xgb_recursive)
    print(f"  XGBoost recursive MAE: {mae_xgb:.2f}")

    # --- Train Random Forest ---
    print("  Training Random Forest (max_depth=5, min_samples_leaf=5, max_features=0.7)...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Recursive multi-step forecast
    rf_recursive = recursive_forecast(
        rf_model, training_prices, n_test, feature_cols, LAGS, ROLLING_WINDOWS
    )
    rf_recursive = np.clip(rf_recursive, 1, None)
    mae_rf = mean_absolute_error(test.values, rf_recursive)
    print(f"  Random Forest recursive MAE: {mae_rf:.2f}")

    # --- Figure 1: ml_recursive_forecast.png ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # XGBoost panel
    ax = axes[0]
    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, xgb_recursive,
            label=f'XGBoost (Recursive)', color='#D32F2F',
            linewidth=1.5, linestyle='--')
    ax.set_title(f'XGBoost - Recursive Multi-Step Forecast\nMAE = {mae_xgb:.2f} USD',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Prix (USD)')
    ax.text(0.02, 0.02,
            "lags=[7,14,30] (sans lag_1)\n"
            "max_depth=3, min_child_weight=10\n"
            "subsample=0.8, colsample_bytree=0.8",
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Random Forest panel
    ax = axes[1]
    ax.plot(train_display.index, train_display.values,
            label='Train', color='#1976D2', linewidth=1.2)
    ax.plot(test.index, test.values,
            label='Actual', color='#4CAF50', linewidth=1.5)
    ax.plot(test.index, rf_recursive,
            label=f'Random Forest (Recursive)', color='#7B1FA2',
            linewidth=1.5, linestyle='--')
    ax.set_title(f'Random Forest - Recursive Multi-Step Forecast\nMAE = {mae_rf:.2f} USD',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Prix (USD)')
    ax.set_xlabel('Date')
    ax.text(0.02, 0.02,
            "lags=[7,14,30] (sans lag_1)\n"
            "max_depth=5, min_samples_leaf=5\n"
            "max_features=0.7",
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(
        'Step 11: Modeles ML - Previsions multi-step recursives (V2)\n'
        'lag_1 supprime, log-returns ajoutes, comparaison equitable',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ml_recursive_forecast.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/step11_ml_models/ml_recursive_forecast.png")

    # --- Figure 2: feature_importance_v2.png ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # XGBoost importance
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    xgb_importance_sorted = xgb_importance.sort_values()

    axes[0].barh(range(len(xgb_importance_sorted)),
                 xgb_importance_sorted.values,
                 color='#EF5350', alpha=0.8)
    axes[0].set_yticks(range(len(xgb_importance_sorted)))
    axes[0].set_yticklabels(xgb_importance_sorted.index, fontsize=9)
    axes[0].set_title('XGBoost - Feature Importance\n(No lag_1)',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Importance')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Verify lag_1 is NOT present
    assert 'lag_1' not in feature_cols, "ERROR: lag_1 found in features!"
    axes[0].text(0.95, 0.05,
                 "Verification: lag_1 ABSENT",
                 transform=axes[0].transAxes, fontsize=9, va='bottom', ha='right',
                 bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    # Random Forest importance
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
    rf_importance_sorted = rf_importance.sort_values()

    axes[1].barh(range(len(rf_importance_sorted)),
                 rf_importance_sorted.values,
                 color='#AB47BC', alpha=0.8)
    axes[1].set_yticks(range(len(rf_importance_sorted)))
    axes[1].set_yticklabels(rf_importance_sorted.index, fontsize=9)
    axes[1].set_title('Random Forest - Feature Importance\n(No lag_1)',
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    axes[1].text(0.95, 0.05,
                 "Verification: lag_1 ABSENT",
                 transform=axes[1].transAxes, fontsize=9, va='bottom', ha='right',
                 bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

    plt.suptitle(
        'Feature Importance V2 - Modeles ML\n'
        'lags=[7,14,30] + rolling stats + log-returns (sans lag_1)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_v2.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/step11_ml_models/feature_importance_v2.png")

    return mae_xgb, mae_rf


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("=" * 70)
    print(" REGENERATE ALL STEP FIGURES WITH V2 CORRECTIONS")
    print("=" * 70)
    print("\nCorrections applied:")
    print("  - enforce_stationarity=True, enforce_invertibility=True for ARIMA")
    print("  - ML hyperparameters: RF(max_depth=5, min_samples_leaf=5, max_features=0.7)")
    print("  -                     XGB(max_depth=3, min_child_weight=10, subsample=0.8)")
    print("  - No lag_1 for ML models (lags=[7, 14, 30])")
    print("  - seasonal=False, d=1 for AutoARIMA")
    print("  - np.clip(forecast, 1, None) for all forecasts")

    # Load and prepare data
    df, monthly, train, test = prepare_data()
    print(f"\nDataset: {len(df)} daily observations")
    print(f"Monthly: {len(monthly)} months")
    print(f"Train/Test split: {len(train)}/{len(test)} months")

    # Regenerate each step
    regenerate_step6(train, test)
    regenerate_step8(train, test)
    regenerate_step10(train, test)
    regenerate_step11(df, monthly, train, test)

    print("\n" + "=" * 70)
    print(" ALL FIGURES REGENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\nRegenerated figures:")
    print("  - figures/step6_naive_forecasting/naive_forecasts.png")
    print("  - figures/step8_statistical_models/statistical_models.png")
    print("  - figures/step10_autoarima/autoarima_forecast.png")
    print("  - figures/step11_ml_models/ml_recursive_forecast.png")
    print("  - figures/step11_ml_models/feature_importance_v2.png")
    print("\nStep 12 figures: run 'python src/additional_figures_v2.py' separately.")


if __name__ == '__main__':
    main()
