"""
Additional Figures for Brent Oil Price Time Series Analysis
Step 12: Model Analysis - Generates 5 additional figures.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data, create_lag_features, evaluate_forecast

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SAVE_DIR = os.path.join(FIGURES_DIR, 'step12_model_analysis')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_ml_data(df):
    """Prepare features and train/test split for ML models."""
    df_feat = create_lag_features(df, target_col='Price',
                                  lags=[1, 7, 14, 30],
                                  rolling_windows=[7, 14, 30])
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c != 'Price']
    X = df_feat[feature_cols]
    y = df_feat['Price']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def train_ml_models(X_train, X_test, y_train, y_test):
    """Train XGBoost and Random Forest and return predictions."""
    xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                      random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    return xgb_model, rf_model, xgb_pred, rf_pred


# ============================================================================
# FIGURE 1: Model Characteristics Comparison Table
# ============================================================================
def figure1_model_characteristics_table():
    """Create a figure showing model characteristics as a table."""
    print("\nGenerating Figure 1: Model Characteristics Table...")

    data = [
        ['Last Value Naive', 'Naive', 'None', 'No', 'No', 'Fast', 'High', 'Baseline comparison'],
        ['Seasonal Naive', 'Naive', 'period=12', 'No', 'Yes', 'Fast', 'High', 'Seasonal baseline'],
        ['Moving Average', 'Naive', 'window=12', 'No', 'No', 'Fast', 'High', 'Smoothed baseline'],
        ['SES', 'Naive', 'alpha=0.3', 'No', 'No', 'Fast', 'High', 'Short-term forecast'],
        ['AR(2)', 'Statistical', 'p=2', 'Yes', 'No', 'Fast', 'High', 'Autoregressive patterns'],
        ['MA(2)', 'Statistical', 'q=2', 'Yes', 'No', 'Fast', 'High', 'Moving average shocks'],
        ['ARMA(2,2)', 'Statistical', 'p=2, q=2', 'Yes', 'No', 'Fast', 'High', 'Combined AR+MA'],
        ['ARIMA(2,1,2)', 'Statistical', 'p=2, d=1, q=2', 'No', 'No', 'Medium', 'Medium', 'Non-stationary series'],
        ['SARIMA', 'Statistical', '(1,1,1)(1,1,0,12)', 'No', 'Yes', 'Medium', 'Medium', 'Seasonal non-stationary'],
        ['AutoARIMA', 'Statistical', 'Auto-selected', 'No', 'Yes', 'Slow', 'Medium', 'Automated model selection'],
        ['XGBoost', 'ML', 'n_est=200, depth=6', 'No', 'No', 'Medium', 'Low', 'Complex nonlinear patterns'],
        ['Random Forest', 'ML', 'n_est=200, depth=15', 'No', 'No', 'Medium', 'Low', 'Ensemble predictions'],
    ]

    columns = ['Model', 'Type', 'Parameters', 'Stationarity\nRequired',
               'Handles\nSeasonality', 'Training\nTime', 'Interpret-\nability', 'Best Use Case']

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#1976D2')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color rows by type
    type_colors = {'Naive': '#E3F2FD', 'Statistical': '#FFF3E0', 'ML': '#E8F5E9'}
    for i, row in enumerate(data, start=1):
        color = type_colors.get(row[1], '#FFFFFF')
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Comparaison des caracteristiques des modeles',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_characteristics_table.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model_characteristics_table.png")


# ============================================================================
# FIGURE 2: Residual Analysis for ML Models
# ============================================================================
def figure2_ml_residual_analysis(y_test, xgb_pred, rf_pred):
    """Create residual analysis plots for XGBoost and Random Forest."""
    print("\nGenerating Figure 2: ML Residual Analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    models = [('XGBoost', xgb_pred, '#D32F2F'),
              ('Random Forest', rf_pred, '#7B1FA2')]

    for row, (name, pred, color) in enumerate(models):
        residuals = y_test.values - pred

        # Residuals over time
        axes[row, 0].plot(y_test.index, residuals, color=color, linewidth=0.7, alpha=0.8)
        axes[row, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[row, 0].set_title(f'{name} - Residus dans le temps', fontsize=11, fontweight='bold')
        axes[row, 0].set_xlabel('Date')
        axes[row, 0].set_ylabel('Residu (USD)')
        axes[row, 0].grid(True, alpha=0.3)

        # Residual distribution (histogram + KDE)
        sns.histplot(residuals, kde=True, bins=40, color=color, ax=axes[row, 1], alpha=0.6)
        axes[row, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[row, 1].set_title(f'{name} - Distribution des residus', fontsize=11, fontweight='bold')
        axes[row, 1].set_xlabel('Residu (USD)')
        axes[row, 1].set_ylabel('Frequence')
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[row, 1].text(0.95, 0.95, f'Moyenne: {mean_res:.3f}\nEcart-type: {std_res:.3f}',
                          transform=axes[row, 1].transAxes, ha='right', va='top',
                          fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # QQ plot
        stats.probplot(residuals, dist="norm", plot=axes[row, 2])
        axes[row, 2].set_title(f'{name} - QQ Plot', fontsize=11, fontweight='bold')
        axes[row, 2].get_lines()[0].set_color(color)
        axes[row, 2].get_lines()[0].set_markersize(3)
        axes[row, 2].get_lines()[1].set_color('black')
        axes[row, 2].grid(True, alpha=0.3)

    plt.suptitle('Analyse des residus - Modeles ML', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'ml_residual_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ml_residual_analysis.png")


# ============================================================================
# FIGURE 3: Predictions vs Actual (Scatter plots)
# ============================================================================
def figure3_predictions_vs_actual(y_test, xgb_pred, rf_pred):
    """Create scatter plots of predicted vs actual values."""
    print("\nGenerating Figure 3: Predictions vs Actual...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [('XGBoost', xgb_pred, '#D32F2F'),
              ('Random Forest', rf_pred, '#7B1FA2')]

    for idx, (name, pred, color) in enumerate(models):
        ax = axes[idx]
        r2 = r2_score(y_test.values, pred)

        ax.scatter(y_test.values, pred, alpha=0.4, s=15, color=color, edgecolors='none')

        # Perfect prediction line
        min_val = min(y_test.values.min(), pred.min())
        max_val = max(y_test.values.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
                label='Prediction parfaite (y=x)')

        ax.set_title(f'{name}\n$R^2$ = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Valeurs reelles (USD)', fontsize=11)
        ax.set_ylabel('Valeurs predites (USD)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Predictions vs Valeurs Reelles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'predictions_vs_actual.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: predictions_vs_actual.png")


# ============================================================================
# FIGURE 4: Learning Curves for ML Models
# ============================================================================
def figure4_learning_curves(X_train, y_train):
    """Plot learning curves for XGBoost and Random Forest."""
    print("\nGenerating Figure 4: Learning Curves...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [
        ('XGBoost', XGBRegressor(n_estimators=100, max_depth=6,
                                  learning_rate=0.05, random_state=42, verbosity=0)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=15,
                                                 random_state=42, n_jobs=-1)),
    ]
    colors = ['#D32F2F', '#7B1FA2']

    train_sizes_frac = np.linspace(0.1, 1.0, 10)

    for idx, (name, model) in enumerate(models):
        ax = axes[idx]

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes_frac,
            cv=5, scoring='r2',
            n_jobs=-1, random_state=42
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color=colors[idx])
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color='#1976D2')

        ax.plot(train_sizes, train_mean, 'o-', color=colors[idx],
                label='Score entrainement', linewidth=2, markersize=5)
        ax.plot(train_sizes, val_mean, 'o-', color='#1976D2',
                label='Score validation (CV=5)', linewidth=2, markersize=5)

        ax.set_title(f'{name} - Courbe d\'apprentissage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Taille de l\'ensemble d\'entrainement', fontsize=11)
        ax.set_ylabel('Score $R^2$', fontsize=11)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.suptitle('Courbes d\'apprentissage - Modeles ML', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'learning_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: learning_curves.png")


# ============================================================================
# FIGURE 5: Error Distribution Comparison
# ============================================================================
def figure5_error_distribution(df):
    """Box plots comparing error distributions across model categories."""
    print("\nGenerating Figure 5: Error Distribution Comparison...")

    # Prepare ML data
    df_feat = create_lag_features(df, target_col='Price',
                                  lags=[1, 7, 14, 30],
                                  rolling_windows=[7, 14, 30])
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c != 'Price']
    X = df_feat[feature_cols]
    y = df_feat['Price']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train ML models
    xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                      random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Monthly data for statistical models
    monthly = df['Price'].resample('MS').mean()
    n_monthly = len(monthly)
    m_split = int(n_monthly * 0.8)
    train_m = monthly[:m_split]
    test_m = monthly[m_split:]
    n_test_m = len(test_m)

    # Naive methods errors
    naive_errors = {}

    # Last Value Naive
    naive_last = np.full(n_test_m, train_m.iloc[-1])
    naive_errors['Last Value'] = np.abs(test_m.values - naive_last)

    # Seasonal Naive
    seasonal_naive = np.array([train_m.iloc[-(12 - i % 12)]
                               for i in range(n_test_m)])
    naive_errors['Seasonal'] = np.abs(test_m.values - seasonal_naive)

    # Moving Average
    ma_value = train_m.iloc[-12:].mean()
    ma_forecast = np.full(n_test_m, ma_value)
    naive_errors['Moving Avg'] = np.abs(test_m.values - ma_forecast)

    # SES
    ses_model = SimpleExpSmoothing(train_m).fit(smoothing_level=0.3, optimized=False)
    ses_forecast = ses_model.forecast(n_test_m)
    naive_errors['SES'] = np.abs(test_m.values - ses_forecast.values)

    # Statistical models errors
    stat_errors = {}

    stat_models = [
        ('AR(2)', (2, 0, 0), None),
        ('ARIMA(2,1,2)', (2, 1, 2), None),
        ('SARIMA', (1, 1, 1), (1, 1, 0, 12)),
    ]

    for name, order, seasonal_order in stat_models:
        try:
            if seasonal_order:
                model = SARIMAX(train_m, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = ARIMA(train_m, order=order,
                              enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_test_m)
            stat_errors[name] = np.abs(test_m.values - forecast.values)
        except Exception as e:
            print(f"  Warning: {name} failed - {e}")
            stat_errors[name] = np.abs(test_m.values - np.full(n_test_m, train_m.mean()))

    # ML model errors (on daily test set)
    ml_errors = {
        'XGBoost': np.abs(y_test.values - xgb_pred),
        'Random Forest': np.abs(y_test.values - rf_pred),
    }

    # Create box plot data
    fig, ax = plt.subplots(figsize=(14, 7))

    # Combine all errors into a format suitable for box plots
    box_data = []
    labels = []
    categories = []

    for name, errors in naive_errors.items():
        box_data.append(errors)
        labels.append(name)
        categories.append('Naive')

    for name, errors in stat_errors.items():
        box_data.append(errors)
        labels.append(name)
        categories.append('Statistical')

    for name, errors in ml_errors.items():
        box_data.append(errors)
        labels.append(name)
        categories.append('ML')

    # Create box plots with color by category
    category_colors = {'Naive': '#42A5F5', 'Statistical': '#FFA726', 'ML': '#66BB6A'}
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, notch=True,
                    medianprops=dict(color='black', linewidth=2))

    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(category_colors[cat])
        patch.set_alpha(0.7)

    # Add legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=color, alpha=0.7, label=cat)
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_patches, fontsize=11, loc='upper right')

    ax.set_title('Distribution des erreurs absolues par modele', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modele', fontsize=12)
    ax.set_ylabel('Erreur absolue (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'error_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: error_distribution.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("=" * 70)
    print(" STEP 12: Additional Model Analysis Figures")
    print("=" * 70)

    ensure_dir(SAVE_DIR)

    # Load data
    df = load_data(os.path.join(DATA_DIR, 'BrentOilPrices.csv'))
    print(f"Dataset loaded: {len(df)} observations")

    # Prepare ML data
    X_train, X_test, y_train, y_test, feature_cols = prepare_ml_data(df)
    print(f"ML data prepared: {len(X_train)} train, {len(X_test)} test samples")

    # Train ML models
    print("\nTraining ML models...")
    xgb_model, rf_model, xgb_pred, rf_pred = train_ml_models(
        X_train, X_test, y_train, y_test)
    print("  XGBoost and Random Forest trained successfully.")

    # Generate all figures
    figure1_model_characteristics_table()
    figure2_ml_residual_analysis(y_test, xgb_pred, rf_pred)
    figure3_predictions_vs_actual(y_test, xgb_pred, rf_pred)
    figure4_learning_curves(X_train, y_train)
    figure5_error_distribution(df)

    print("\n" + "=" * 70)
    print(" Step 12 complete - All additional figures saved to:")
    print(f"   {SAVE_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
