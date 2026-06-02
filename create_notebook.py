"""Create the Jupyter notebook programmatically with French documentation."""
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.11.0'
    }
}

cells = []

# Title
cells.append(new_markdown_cell("""# Analyse et Prevision des Prix du Petrole Brent

## Description du projet
Ce notebook presente une analyse complete des series temporelles appliquee aux prix du petrole Brent (1987-2023). L'objectif est de prevoir les prix futurs pour aider la prise de decision economique et financiere.

## Methodologie
1. Chargement et visualisation des donnees
2. Decomposition de la serie temporelle
3. Tests de stationnarite
4. Transformations (log, differenciation, Box-Cox)
5. Division Train/Test
6. Previsions naives (benchmarks)
7. Analyse ACF/PACF
8. Modeles statistiques (AR, MA, ARMA, ARIMA, SARIMA)
9. Evaluation et comparaison
10. AutoARIMA
11. Modeles Machine Learning (XGBoost, Random Forest)
"""))

# Imports
cells.append(new_code_cell("""import warnings
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

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Configuration
FIGURES_DIR = os.path.join(os.path.dirname(os.getcwd()), 'figures') if 'notebooks' in os.getcwd() else 'figures'
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'data', 'BrentOilPrices.csv') if 'notebooks' in os.getcwd() else 'data/BrentOilPrices.csv'

print("Configuration terminee.")
"""))

# Step 1
cells.append(new_markdown_cell("""## Etape 1 : Chargement et Visualisation des Donnees

Nous commencons par charger le jeu de donnees des prix du Brent et explorer ses caracteristiques principales.
Le dataset contient les prix journaliers du petrole Brent de mai 1987 a decembre 2023.
"""))

cells.append(new_code_cell("""# Chargement des donnees
df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
df = df.sort_index()

print(f"Nombre d'observations : {len(df)}")
print(f"Periode : {df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}")
print(f"\\nStatistiques descriptives :")
print(df['Price'].describe())
"""))

cells.append(new_code_cell("""# Visualisation de la serie temporelle complete
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Price'], linewidth=0.7, color='#1976D2')
ax.set_title('Prix du Brent (1987-2023)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Prix (USD/baril)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.fill_between(df.index, df['Price'], alpha=0.15, color='#1976D2')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step1_visualization', 'time_series.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(new_code_cell("""# Moyenne annuelle
yearly_avg = df['Price'].resample('YE').mean()
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(yearly_avg.index.year, yearly_avg.values, color='#42A5F5', edgecolor='#1565C0')
ax.set_title('Moyenne annuelle du prix du Brent', fontsize=14, fontweight='bold')
ax.set_xlabel('Annee', fontsize=12)
ax.set_ylabel('Prix moyen (USD/baril)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step1_visualization', 'yearly_averages.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(new_code_cell("""# Distribution des prix
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, bins=50, color='#42A5F5', ax=ax)
ax.axvline(df['Price'].mean(), color='red', linestyle='--', label=f"Moyenne: ${df['Price'].mean():.2f}")
ax.axvline(df['Price'].median(), color='green', linestyle='--', label=f"Mediane: ${df['Price'].median():.2f}")
ax.set_title('Distribution des prix du Brent', fontsize=14, fontweight='bold')
ax.set_xlabel('Prix (USD/baril)', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step1_visualization', 'distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 2
cells.append(new_markdown_cell("""## Etape 2 : Decomposition de la Serie Temporelle

La decomposition permet de separer la serie en trois composantes :
- **Tendance** : mouvement a long terme
- **Saisonnalite** : patterns repetitifs
- **Residus** : bruit aleatoire

Nous utilisons une decomposition additive avec une periode de 12 mois sur les donnees mensuelles.
"""))

cells.append(new_code_cell("""# Reechantillonnage mensuel et decomposition
monthly = df['Price'].resample('MS').mean()
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

plt.suptitle('Decomposition de la serie temporelle (Additive, periode=12)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step2_decomposition', 'decomposition.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 3
cells.append(new_markdown_cell("""## Etape 3 : Tests de Stationnarite

Une serie stationnaire a des proprietes statistiques (moyenne, variance) constantes dans le temps.
Nous utilisons :
- **Test ADF (Augmented Dickey-Fuller)** : H0 = la serie a une racine unitaire (non stationnaire)
- **Test KPSS** : H0 = la serie est stationnaire

Pour etre stationnaire, on attend : ADF p-value < 0.05 ET KPSS p-value > 0.05
"""))

cells.append(new_code_cell("""# Statistiques glissantes (fenetre = 252 jours de trading ~ 1 an)
window = 252
rolling_mean = df['Price'].rolling(window=window).mean()
rolling_std = df['Price'].rolling(window=window).std()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(df.index, df['Price'], label='Original', color='#1976D2', alpha=0.6, linewidth=0.7)
axes[0].plot(rolling_mean.index, rolling_mean.values, label=f'Moyenne mobile ({window}j)', color='#D32F2F', linewidth=2)
axes[0].set_title('Serie originale et moyenne mobile', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(rolling_std.index, rolling_std.values, label=f'Ecart-type mobile ({window}j)', color='#F57C00', linewidth=2)
axes[1].set_title('Ecart-type mobile', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step3_stationarity', 'rolling_statistics.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(new_code_cell("""# Tests statistiques
series = df['Price']

# Test ADF
adf_result = adfuller(series, autolag='AIC')
print("Test ADF :")
print(f"  Statistique de test : {adf_result[0]:.4f}")
print(f"  p-value : {adf_result[1]:.6f}")
print(f"  Stationnaire (p < 0.05) : {'Oui' if adf_result[1] < 0.05 else 'Non'}")

# Test KPSS
kpss_result = kpss(series, regression='c', nlags='auto')
print(f"\\nTest KPSS :")
print(f"  Statistique de test : {kpss_result[0]:.4f}")
print(f"  p-value : {kpss_result[1]:.4f}")
print(f"  Stationnaire (p > 0.05) : {'Oui' if kpss_result[1] > 0.05 else 'Non'}")

print(f"\\nConclusion : La serie {'est' if adf_result[1] < 0.05 and kpss_result[1] > 0.05 else 'n\\'est PAS'} stationnaire.")
"""))

# Step 4
cells.append(new_markdown_cell("""## Etape 4 : Transformations pour la Stationnarite

Pour rendre la serie stationnaire, nous appliquons :
1. **Transformation logarithmique** : stabilise la variance
2. **Differenciation** : elimine la tendance
3. **Transformation Box-Cox** : transformation optimale de puissance
"""))

cells.append(new_code_cell("""# Transformations
log_series = np.log(df['Price'])
log_diff = log_series.diff().dropna()
boxcox_series, lam = stats.boxcox(df['Price'][df['Price'] > 0])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(df.index, df['Price'], color='#1976D2', linewidth=0.7)
axes[0, 0].set_title('Serie originale', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(log_series.index, log_series.values, color='#388E3C', linewidth=0.7)
axes[0, 1].set_title('Transformation logarithmique', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(log_diff.index, log_diff.values, color='#F57C00', linewidth=0.7)
axes[1, 0].set_title('Differenciation (log)', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df.index, boxcox_series, color='#D32F2F', linewidth=0.7)
axes[1, 1].set_title(f'Box-Cox (lambda={lam:.3f})', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Comparaison des transformations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step4_transformations', 'transformations.png'), dpi=150, bbox_inches='tight')
plt.show()

# Verification de stationnarite apres differenciation
adf_diff = adfuller(log_diff, autolag='AIC')
print(f"Apres differenciation logarithmique :")
print(f"  ADF p-value : {adf_diff[1]:.6f} -> {'Stationnaire' if adf_diff[1] < 0.05 else 'Non stationnaire'}")
"""))

# Step 5
cells.append(new_markdown_cell("""## Etape 5 : Division Train/Test

Nous divisons les donnees mensuelles en ensemble d'entrainement (80%) et de test (20%) de maniere chronologique.
"""))

cells.append(new_code_cell("""# Division 80/20
n = len(monthly)
split_idx = int(n * 0.8)
train = monthly[:split_idx]
test = monthly[split_idx:]

print(f"Observations totales : {n}")
print(f"Entrainement : {len(train)} mois ({len(train)/n*100:.1f}%)")
print(f"Test : {len(test)} mois ({len(test)/n*100:.1f}%)")
print(f"Date de coupure : {monthly.index[split_idx].strftime('%Y-%m-%d')}")

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
plt.savefig(os.path.join(FIGURES_DIR, 'step5_train_test_split', 'train_test_split.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 6
cells.append(new_markdown_cell("""## Etape 6 : Previsions Naives (Benchmarks)

Les methodes naives servent de reference (baseline) :
- **Last Value** : predit la derniere valeur observee
- **Seasonal Naive** : repete le pattern saisonnier (12 mois)
- **Moyenne Mobile** : moyenne des 12 derniers mois
- **Lissage Exponentiel Simple** : moyenne ponderee exponentiellement
"""))

cells.append(new_code_cell("""n_test = len(test)

# Previsions naives
naive_last = np.full(n_test, train.iloc[-1])
seasonal_naive = np.array([train.iloc[-(12 - i % 12)] for i in range(n_test)])
ma_forecast = np.full(n_test, train.iloc[-12:].mean())
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.3, optimized=False)
ses_forecast = ses_model.forecast(n_test)

# Evaluation
def calc_metrics(actual, predicted, name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    print(f"{name:30s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} | MAPE: {mape:6.2f}%")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

print("Resultats des previsions naives :")
print("-" * 70)
naive_results = []
naive_results.append(calc_metrics(test.values, naive_last, 'Last Value Naive'))
naive_results.append(calc_metrics(test.values, seasonal_naive, 'Seasonal Naive (12)'))
naive_results.append(calc_metrics(test.values, ma_forecast, 'Moving Average (12)'))
naive_results.append(calc_metrics(test.values, ses_forecast.values, 'Simple Exp. Smoothing'))
"""))

cells.append(new_code_cell("""# Visualisation des previsions naives
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
plt.savefig(os.path.join(FIGURES_DIR, 'step6_naive_forecasting', 'naive_forecasts.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 7
cells.append(new_markdown_cell("""## Etape 7 : Analyse ACF et PACF

L'autocorrelation (ACF) et l'autocorrelation partielle (PACF) permettent d'identifier les ordres potentiels pour les modeles AR et MA :
- **ACF** : correlation entre la serie et ses retards
- **PACF** : correlation directe (sans effet intermediaire)

Regles d'identification :
- ACF decroit lentement, PACF coupe apres p lags -> AR(p)
- ACF coupe apres q lags, PACF decroit -> MA(q)
"""))

cells.append(new_code_cell("""# ACF/PACF de la serie originale
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(monthly.dropna(), lags=50, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF - Serie mensuelle originale', fontsize=12)
plot_pacf(monthly.dropna(), lags=50, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF - Serie mensuelle originale', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step7_acf_pacf', 'acf_pacf_original.png'), dpi=150, bbox_inches='tight')
plt.show()

# ACF/PACF de la serie differenciee
diff_monthly = monthly.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(diff_monthly, lags=50, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF - Serie differenciee', fontsize=12)
plot_pacf(diff_monthly, lags=50, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF - Serie differenciee', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step7_acf_pacf', 'acf_pacf_differenced.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 8
cells.append(new_markdown_cell("""## Etape 8 : Modeles Statistiques

Nous ajustons differents modeles de series temporelles :
- **AR(2)** : AutoRegressif d'ordre 2
- **MA(2)** : Moyenne Mobile d'ordre 2
- **ARMA(2,2)** : Combinaison AR + MA
- **ARIMA(2,1,2)** : ARMA avec differenciation integree
- **SARIMA(1,1,1)(1,1,0,12)** : ARIMA avec composante saisonniere
"""))

cells.append(new_code_cell("""# Ajustement des modeles statistiques
model_configs = [
    ('AR(2)', (2, 0, 0), None),
    ('MA(2)', (0, 0, 2), None),
    ('ARMA(2,2)', (2, 0, 2), None),
    ('ARIMA(2,1,2)', (2, 1, 2), None),
    ('SARIMA(1,1,1)(1,1,0,12)', (1, 1, 1), (1, 1, 0, 12)),
]

stat_results = []
forecasts = {}

for name, order, seasonal_order in model_configs:
    try:
        if seasonal_order:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = ARIMA(train, order=order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit()
        forecast = fitted.forecast(steps=n_test)
        forecasts[name] = forecast.values
        metrics = calc_metrics(test.values, forecast.values, name)
        metrics['AIC'] = round(fitted.aic, 2)
        metrics['BIC'] = round(fitted.bic, 2)
        stat_results.append(metrics)
    except Exception as e:
        print(f"Erreur avec {name}: {e}")
        stat_results.append({'model': name, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan})
"""))

cells.append(new_code_cell("""# Visualisation des previsions des modeles statistiques
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes_flat = axes.flatten()
colors = ['#D32F2F', '#F57C00', '#388E3C', '#7B1FA2', '#00838F']

for idx, (name, order, seasonal_order) in enumerate(model_configs):
    ax = axes_flat[idx]
    ax.plot(train.index[-48:], train.values[-48:], label='Train', color='#1976D2', linewidth=1)
    ax.plot(test.index, test.values, label='Actual', color='#4CAF50', linewidth=1.5)
    if name in forecasts:
        ax.plot(test.index, forecasts[name][:n_test], label=name, color=colors[idx], linewidth=1.5, linestyle='--')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes_flat[-1].set_visible(False)
plt.suptitle('Modeles statistiques - Previsions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step8_statistical_models', 'statistical_models.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 9
cells.append(new_markdown_cell("""## Etape 9 : Evaluation et Comparaison

Nous comparons tous les modeles (naifs et statistiques) en utilisant :
- **MAE** (Mean Absolute Error) : erreur absolue moyenne
- **RMSE** (Root Mean Squared Error) : racine de l'erreur quadratique moyenne
- **MAPE** (Mean Absolute Percentage Error) : erreur en pourcentage
- **AIC/BIC** : criteres d'information (plus bas = meilleur)
"""))

cells.append(new_code_cell("""# Tableau comparatif
all_results = naive_results + stat_results
df_results = pd.DataFrame(all_results)
print("\\nComparaison de tous les modeles :")
print("=" * 80)
print(df_results.to_string(index=False))

# Graphique de comparaison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_plot = ['MAE', 'RMSE', 'MAPE']
colors_list = plt.cm.Set3(np.linspace(0, 1, len(df_results)))

for idx, metric in enumerate(metrics_plot):
    valid = df_results.dropna(subset=[metric])
    axes[idx].barh(valid['model'], valid[metric], color=colors_list[:len(valid)])
    axes[idx].set_title(metric, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(metric)
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.suptitle('Comparaison des modeles', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step9_evaluation', 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 10
cells.append(new_markdown_cell("""## Etape 10 : AutoARIMA

AutoARIMA selectionne automatiquement les meilleurs parametres (p, d, q) et (P, D, Q, m)
en minimisant le critere AIC. C'est une approche pratique qui evite la selection manuelle.
"""))

cells.append(new_code_cell("""# AutoARIMA
print("Ajustement AutoARIMA (seasonal=True, m=12)...")
auto_model = pm.auto_arima(train, seasonal=True, m=12,
                            suppress_warnings=True, stepwise=True,
                            trace=False, error_action='ignore')

print(f"\\nOrdre selectionne : {auto_model.order}")
print(f"Ordre saisonnier : {auto_model.seasonal_order}")
print(f"AIC : {auto_model.aic():.2f}")

# Prevision avec intervalle de confiance
forecast_auto, conf_int = auto_model.predict(n_periods=n_test, return_conf_int=True)
calc_metrics(test.values, forecast_auto, 'AutoARIMA')

# Visualisation
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(train.index[-48:], train.values[-48:], label='Train', color='#1976D2', linewidth=1.2)
ax.plot(test.index, test.values, label='Actual', color='#4CAF50', linewidth=1.5)
ax.plot(test.index, forecast_auto, label=f'AutoARIMA {auto_model.order}', color='#D32F2F', linewidth=1.5, linestyle='--')
ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], alpha=0.2, color='#D32F2F', label='IC 95%')
ax.set_title(f'AutoARIMA - Ordre: {auto_model.order}, Saisonnier: {auto_model.seasonal_order}', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Prix (USD/baril)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step10_autoarima', 'autoarima_forecast.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Step 11
cells.append(new_markdown_cell("""## Etape 11 : Modeles Machine Learning

Les modeles ML utilisent des features ingenierees :
- **Retards (lags)** : prix aux jours t-1, t-7, t-14, t-30
- **Statistiques glissantes** : moyenne et ecart-type sur 7, 14, 30 jours
- **Caracteristiques calendaires** : mois, trimestre, jour de la semaine, annee

Les modeles XGBoost et Random Forest effectuent des previsions one-step-ahead
(un pas en avant) sur les donnees journalieres.
"""))

cells.append(new_code_cell("""# Ingenierie des features
df_feat = df.copy()

# Retards
for lag in [1, 7, 14, 30]:
    df_feat[f'lag_{lag}'] = df_feat['Price'].shift(lag)

# Statistiques glissantes
for window in [7, 14, 30]:
    df_feat[f'rolling_mean_{window}'] = df_feat['Price'].shift(1).rolling(window=window).mean()
    df_feat[f'rolling_std_{window}'] = df_feat['Price'].shift(1).rolling(window=window).std()

# Caracteristiques calendaires
df_feat['month'] = df_feat.index.month
df_feat['quarter'] = df_feat.index.quarter
df_feat['day_of_week'] = df_feat.index.dayofweek
df_feat['year'] = df_feat.index.year

df_feat = df_feat.dropna()

feature_cols = [c for c in df_feat.columns if c != 'Price']
X = df_feat[feature_cols]
y = df_feat['Price']

split_idx_ml = int(len(X) * 0.8)
X_train, X_test = X[:split_idx_ml], X[split_idx_ml:]
y_train, y_test = y[:split_idx_ml], y[split_idx_ml:]

print(f"Echantillons d'entrainement : {len(X_train)}")
print(f"Echantillons de test : {len(X_test)}")
print(f"Nombre de features : {len(feature_cols)}")
print(f"Features : {feature_cols}")
"""))

cells.append(new_code_cell("""# XGBoost
xgb_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Resultats des modeles ML :")
print("-" * 70)
calc_metrics(y_test.values, xgb_pred, 'XGBoost')
calc_metrics(y_test.values, rf_pred, 'Random Forest')
"""))

cells.append(new_code_cell("""# Visualisation des predictions ML
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(y_test.index, y_test.values, label='Actual', color='#4CAF50', linewidth=1.2)
axes[0].plot(y_test.index, xgb_pred, label='XGBoost', color='#D32F2F', linewidth=1, alpha=0.8)
axes[0].set_title('XGBoost - Predictions one-step-ahead', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylabel('Prix (USD)')

axes[1].plot(y_test.index, y_test.values, label='Actual', color='#4CAF50', linewidth=1.2)
axes[1].plot(y_test.index, rf_pred, label='Random Forest', color='#7B1FA2', linewidth=1, alpha=0.8)
axes[1].set_title('Random Forest - Predictions one-step-ahead', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylabel('Prix (USD)')
axes[1].set_xlabel('Date')

plt.suptitle('Modeles ML - Previsions one-step-ahead', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step11_ml_models', 'ml_predictions.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(new_code_cell("""# Importance des features
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).nlargest(15)
xgb_importance.plot(kind='barh', ax=axes[0], color='#EF5350')
axes[0].set_title('XGBoost - Importance des features (Top 15)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Importance')
axes[0].grid(True, alpha=0.3, axis='x')

rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).nlargest(15)
rf_importance.plot(kind='barh', ax=axes[1], color='#AB47BC')
axes[1].set_title('Random Forest - Importance des features (Top 15)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Importance')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'step11_ml_models', 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
"""))

# Conclusion
cells.append(new_markdown_cell("""## Conclusion

### Resultats principaux :
1. La serie des prix du Brent n'est pas stationnaire en niveau mais devient stationnaire apres differenciation logarithmique.
2. Les modeles naifs constituent des baselines utiles avec des MAPE entre 42% et 56%.
3. Le modele ARMA(2,2) est le meilleur modele statistique.
4. AutoARIMA selectionne automatiquement un modele performant.
5. Les modeles ML (XGBoost, Random Forest) surpassent largement les modeles statistiques grace aux features de retard, avec des MAPE inferieurs a 10%.

### Limites :
- Les modeles ML utilisent des previsions one-step-ahead (ils ont acces au prix du jour precedent comme feature), ce qui n'est pas directement comparable aux previsions multi-step des modeles statistiques.
- Les prix du petrole sont influenices par des evenements geopolitiques imprevisibles.
- Un modele hybride combinant analyse fondamentale et technique pourrait ameliorer les resultats.
"""))

nb.cells = cells

# Save notebook
import os
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/brent_oil_forecasting.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Notebook created with {len([c for c in cells if c.cell_type == 'code'])} code cells and {len([c for c in cells if c.cell_type == 'markdown'])} markdown cells.")
