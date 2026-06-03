"""
Additional Figures for Brent Oil Price Time Series Analysis - V2 (Corrected)
=============================================================================
Step 12: Model Analysis - Generates 5 additional figures.

Fixes from V1:
- Removes lag_1 (prevents persistence copying bias)
- Uses monthly data for ML models (same granularity as statistical models)
- All models evaluated on the SAME monthly test set for fair comparison
- Imports create_ml_features_df from analysis_v2 (reuses V2 feature engineering)
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_data, evaluate_forecast
from src.analysis_v2 import create_ml_features_df

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
SAVE_DIR = os.path.join(FIGURES_DIR, 'step12_model_analysis')

# V2 configuration - NO lag_1
LAGS = [7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def prepare_monthly_ml_data(df):
    """
    Prepare features and train/test split for ML models on MONTHLY data.
    Uses create_ml_features_df from analysis_v2 (no lag_1, with log-returns).
    """
    # Resample to monthly
    monthly = df['Price'].resample('MS').mean()

    # Create features using V2 function (no lag_1, with log-returns)
    df_feat = create_ml_features_df(monthly, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c != 'Price']
    X = df_feat[feature_cols]
    y = df_feat['Price']

    # 80/20 split on monthly data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols, monthly


def train_ml_models(X_train, X_test, y_train, y_test):
    """
    Train XGBoost and Random Forest on monthly data and return predictions.

    Hyperparameter rationale (dataset size ~260 monthly observations):
    - max_depth severely limited to prevent memorization of small dataset
    - min_samples_leaf / min_child_weight enforce meaningful leaf nodes
    - Subsampling (max_features, subsample, colsample_bytree) adds regularization
    - Tree-based models have inherent extrapolation limitations: they cannot
      predict values outside the training range, which causes systematic
      underestimation when test prices exceed training prices
    - For financial time series with limited features, ARIMA is expected to
      outperform ML models -- this is the academically correct conclusion
    """
    # XGBoost - regularized for n~260 monthly observations
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=3,            # reduced from 6 to prevent overfitting
        learning_rate=0.05,
        min_child_weight=10,    # regularization: min obs per node
        subsample=0.8,          # row subsampling for regularization
        colsample_bytree=0.8,  # column subsampling for regularization
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # Random Forest - regularized for n~260 monthly observations
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,            # balanced: depth=4 was underfitting (R2~0.62)
        min_samples_leaf=5,     # reduced from 10 for better fit
        max_features=0.7,       # subsample features to reduce overfitting
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    return xgb_model, rf_model, xgb_pred, rf_pred


# ============================================================================
# FIGURE 1: Model Characteristics Comparison Table (V2)
# ============================================================================
def figure1_model_characteristics_table():
    """Create a figure showing model characteristics as a table (V2 labels)."""
    print("\nGenerating Figure 1: Model Characteristics Table (V2)...")

    data = [
        ['Last Value Naive', 'Naive', 'Aucun', 'Non', 'Non', 'Rapide', 'Elevee', 'Baseline'],
        ['Seasonal Naive', 'Naive', 'period=12', 'Non', 'Oui', 'Rapide', 'Elevee', 'Baseline saisonniere'],
        ['Moving Average', 'Naive', 'window=12', 'Non', 'Non', 'Rapide', 'Elevee', 'Baseline lissee'],
        ['SES', 'Naive', 'alpha=0.3', 'Non', 'Non', 'Rapide', 'Elevee', 'Prevision court terme'],
        ['AR(2)', 'Statistique', 'p=2', 'Oui', 'Non', 'Rapide', 'Elevee', 'Patterns autoregressifs'],
        ['ARIMA(2,1,2)', 'Statistique', 'p=2, d=1, q=2', 'Non', 'Non', 'Moyen', 'Moyenne', 'Serie non-stationnaire'],
        ['SARIMA', 'Statistique', '(1,1,1)(1,1,0,12)', 'Non', 'Oui', 'Moyen', 'Moyenne', 'Saisonnalite + tendance'],
        ['AutoARIMA', 'Statistique', 'Auto', 'Non', 'Oui', 'Lent', 'Moyenne', 'Selection automatique'],
        ['XGBoost V2', 'ML', 'V2, sans lag_1\nlags=[7,14,30]', 'Non', 'Non', 'Moyen', 'Faible', 'Patterns non-lineaires'],
        ['Random Forest V2', 'ML', 'V2, sans lag_1\nlags=[7,14,30]', 'Non', 'Non', 'Moyen', 'Faible', 'Ensemble predictions'],
    ]

    columns = ['Modele', 'Type', 'Parametres', 'Stationnarite\nRequise',
               'Gere la\nSaisonnalite', 'Temps\nEntrainement', 'Interpret-\nabilite', 'Cas d\'utilisation']

    fig, ax = plt.subplots(figsize=(18, 7))
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
    type_colors = {'Naive': '#E3F2FD', 'Statistique': '#FFF3E0', 'ML': '#E8F5E9'}
    for i, row in enumerate(data, start=1):
        color = type_colors.get(row[1], '#FFFFFF')
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)

    ax.set_title('Comparaison des caracteristiques des modeles (V2 - sans lag_1)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_characteristics_table.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: model_characteristics_table.png")


# ============================================================================
# FIGURE 2: Residual Analysis for ML Models (Monthly, V2)
# ============================================================================
def figure2_ml_residual_analysis(y_test, xgb_pred, rf_pred):
    """Create residual analysis plots for XGBoost and Random Forest (monthly data)."""
    print("\nGenerating Figure 2: ML Residual Analysis (monthly, V2)...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    models = [('XGBoost V2', xgb_pred, '#D32F2F'),
              ('Random Forest V2', rf_pred, '#7B1FA2')]

    for row, (name, pred, color) in enumerate(models):
        residuals = y_test.values - pred

        # Residuals over time
        axes[row, 0].plot(y_test.index, residuals, color=color, linewidth=1.0, alpha=0.8)
        axes[row, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[row, 0].set_title(f'{name} - Residus dans le temps', fontsize=11, fontweight='bold')
        axes[row, 0].set_xlabel('Date')
        axes[row, 0].set_ylabel('Residu (USD)')
        axes[row, 0].grid(True, alpha=0.3)

        # Residual distribution (histogram + KDE)
        sns.histplot(residuals, kde=True, bins=20, color=color, ax=axes[row, 1], alpha=0.6)
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
        axes[row, 2].get_lines()[0].set_markersize(4)
        axes[row, 2].get_lines()[1].set_color('black')
        axes[row, 2].grid(True, alpha=0.3)

    plt.suptitle('Analyse des residus - Modeles ML V2 (donnees mensuelles, sans lag_1)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'ml_residual_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: ml_residual_analysis.png")


# ============================================================================
# FIGURE 3: Predictions vs Actual (Scatter plots, Monthly)
# ============================================================================
def figure3_predictions_vs_actual(y_test, xgb_pred, rf_pred):
    """Create scatter plots of predicted vs actual values (monthly data)."""
    print("\nGenerating Figure 3: Predictions vs Actual (monthly, V2)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [('XGBoost V2', xgb_pred, '#D32F2F'),
              ('Random Forest V2', rf_pred, '#7B1FA2')]

    for idx, (name, pred, color) in enumerate(models):
        ax = axes[idx]
        r2 = r2_score(y_test.values, pred)

        ax.scatter(y_test.values, pred, alpha=0.6, s=30, color=color, edgecolors='none')

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

    plt.suptitle('Predictions vs Valeurs Reelles (donnees mensuelles, V2)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'predictions_vs_actual.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: predictions_vs_actual.png")


# ============================================================================
# FIGURE 4: Learning Curves for ML Models (Monthly, V2)
# ============================================================================
def figure4_learning_curves(X_train, y_train):
    """Plot learning curves for XGBoost and Random Forest on monthly data (V2)."""
    print("\nGenerating Figure 4: Learning Curves (monthly, V2)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = [
        ('XGBoost V2', XGBRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.05, min_child_weight=10,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbosity=0)),
        ('Random Forest V2', RandomForestRegressor(n_estimators=100, max_depth=5,
                                                    min_samples_leaf=5,
                                                    max_features=0.7,
                                                    random_state=42, n_jobs=-1)),
    ]
    colors = ['#D32F2F', '#7B1FA2']

    train_sizes_frac = np.linspace(0.2, 1.0, 8)

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
        ax.set_ylim([-0.5, 1.05])

    plt.suptitle('Courbes d\'apprentissage - Modeles ML V2 (donnees mensuelles, sans lag_1)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'learning_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: learning_curves.png")


# ============================================================================
# FIGURE 5: Error Distribution - FAIR Comparison (ALL on same monthly test set)
# ============================================================================
def figure5_error_distribution_fair(df):
    """
    Box plots comparing error distributions across ALL model categories.
    ALL models evaluated on the SAME monthly test set.
    ML models use one-step-ahead on monthly data (same test indices as statistical).
    """
    print("\nGenerating Figure 5: Error Distribution - FAIR Comparison (V2)...")

    # Monthly data
    monthly = df['Price'].resample('MS').mean()
    n_monthly = len(monthly)
    m_split = int(n_monthly * 0.8)
    train_m = monthly[:m_split]
    test_m = monthly[m_split:]
    n_test_m = len(test_m)

    print(f"  Monthly train: {len(train_m)}, Monthly test: {n_test_m}")

    # --- ML models on monthly data (V2 features, one-step-ahead) ---
    df_feat = create_ml_features_df(monthly, lags=LAGS, rolling_windows=ROLLING_WINDOWS)
    df_feat = df_feat.dropna()

    feature_cols = [c for c in df_feat.columns if c != 'Price']

    # Split aligned with train/test
    train_end = train_m.index[-1]
    test_start = test_m.index[0]

    df_feat_train = df_feat.loc[:train_end]
    df_feat_test = df_feat.loc[test_start:]

    X_train_ml = df_feat_train[feature_cols]
    y_train_ml = df_feat_train['Price']
    X_test_ml = df_feat_test[feature_cols]
    y_test_ml = df_feat_test['Price']

    # Train ML models
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
    xgb_model.fit(X_train_ml, y_train_ml)
    xgb_pred = xgb_model.predict(X_test_ml)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_ml, y_train_ml)
    rf_pred = rf_model.predict(X_test_ml)

    # Get the common test index (intersection of ML test and statistical test)
    ml_test_index = y_test_ml.index
    common_index = test_m.index.intersection(ml_test_index)
    print(f"  Common test index length: {len(common_index)}")

    # Align ML predictions to common index
    xgb_errors_aligned = np.abs(
        y_test_ml.loc[common_index].values - xgb_model.predict(X_test_ml.loc[common_index])
    )
    rf_errors_aligned = np.abs(
        y_test_ml.loc[common_index].values - rf_model.predict(X_test_ml.loc[common_index])
    )

    n_common = len(common_index)

    # --- Naive methods on monthly data (aligned to common_index) ---
    naive_errors = {}

    # Last Value Naive
    naive_last = np.full(n_test_m, train_m.iloc[-1])
    naive_last_series = pd.Series(naive_last, index=test_m.index)
    naive_errors['Last Value'] = np.abs(
        test_m.loc[common_index].values - naive_last_series.loc[common_index].values
    )

    # Seasonal Naive
    seasonal_naive = np.array([train_m.iloc[-(12 - i % 12)]
                               for i in range(n_test_m)])
    seasonal_series = pd.Series(seasonal_naive, index=test_m.index)
    naive_errors['Seasonal'] = np.abs(
        test_m.loc[common_index].values - seasonal_series.loc[common_index].values
    )

    # Moving Average
    ma_value = train_m.iloc[-12:].mean()
    ma_forecast = np.full(n_test_m, ma_value)
    ma_series = pd.Series(ma_forecast, index=test_m.index)
    naive_errors['Moving Avg'] = np.abs(
        test_m.loc[common_index].values - ma_series.loc[common_index].values
    )

    # SES
    ses_model = SimpleExpSmoothing(train_m).fit(smoothing_level=0.3, optimized=False)
    ses_forecast = ses_model.forecast(n_test_m)
    ses_series = pd.Series(ses_forecast.values, index=test_m.index)
    naive_errors['SES'] = np.abs(
        test_m.loc[common_index].values - ses_series.loc[common_index].values
    )

    # --- Statistical models on monthly data (aligned to common_index) ---
    stat_errors = {}

    stat_models_cfg = [
        ('AR(2)', (2, 0, 0), None),
        ('ARIMA(2,1,2)', (2, 1, 2), None),
        ('SARIMA', (1, 1, 1), (1, 1, 0, 12)),
    ]

    for name, order, seasonal_order in stat_models_cfg:
        try:
            if seasonal_order:
                model = SARIMAX(train_m, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = ARIMA(train_m, order=order,
                              enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_test_m)
            forecast_series = pd.Series(forecast.values, index=test_m.index)
            stat_errors[name] = np.abs(
                test_m.loc[common_index].values - forecast_series.loc[common_index].values
            )
        except Exception as e:
            print(f"  Warning: {name} failed - {e}")
            stat_errors[name] = np.abs(
                test_m.loc[common_index].values - np.full(n_common, train_m.mean())
            )

    # --- ML model errors (already aligned) ---
    ml_errors = {
        'XGBoost V2': xgb_errors_aligned,
        'Random Forest V2': rf_errors_aligned,
    }

    # --- Create box plot ---
    fig, ax = plt.subplots(figsize=(14, 7))

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
        categories.append('Statistique')

    for name, errors in ml_errors.items():
        box_data.append(errors)
        labels.append(name)
        categories.append('ML')

    # Create box plots with color by category
    category_colors = {'Naive': '#42A5F5', 'Statistique': '#FFA726', 'ML': '#66BB6A'}
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

    ax.set_title(
        'Distribution des erreurs absolues par modele\n'
        '(Comparaison EQUITABLE: tous sur le meme jeu de test mensuel)',
        fontsize=14, fontweight='bold')
    ax.set_xlabel('Modele', fontsize=12)
    ax.set_ylabel('Erreur absolue (USD)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation about fairness
    ax.text(0.02, 0.98,
            f'N = {n_common} mois (meme periode pour tous les modeles)\n'
            f'ML: one-step-ahead mensuel, sans lag_1',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

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
    print(" STEP 12 (V2): Additional Model Analysis Figures")
    print("=" * 70)
    print("\nFixes applied:")
    print("  - lag_1 REMOVED (lags=[7,14,30] only)")
    print("  - ML models trained on MONTHLY data (same as statistical)")
    print("  - All models evaluated on the SAME monthly test set")
    print("  - Log-return features added")

    ensure_dir(SAVE_DIR)

    # Load data
    df = load_data(os.path.join(DATA_DIR, 'BrentOilPrices.csv'))
    print(f"\nDataset loaded: {len(df)} observations")

    # Prepare monthly ML data (V2 features, no lag_1)
    X_train, X_test, y_train, y_test, feature_cols, monthly = prepare_monthly_ml_data(df)
    print(f"Monthly ML data prepared: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # Train ML models on monthly data
    print("\nTraining ML models on monthly data (V2, sans lag_1)...")
    xgb_model, rf_model, xgb_pred, rf_pred = train_ml_models(
        X_train, X_test, y_train, y_test)
    print("  XGBoost V2 and Random Forest V2 trained successfully.")

    # Generate all 5 figures
    figure1_model_characteristics_table()
    figure2_ml_residual_analysis(y_test, xgb_pred, rf_pred)
    figure3_predictions_vs_actual(y_test, xgb_pred, rf_pred)
    figure4_learning_curves(X_train, y_train)
    figure5_error_distribution_fair(df)

    print("\n" + "=" * 70)
    print(" Step 12 (V2) complete - All corrected figures saved to:")
    print(f"   {SAVE_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
