# Analyse et Prevision des Prix du Petrole Brent

## Description

Ce projet presente une analyse complete des series temporelles appliquee aux prix du petrole brut Brent, couvrant la periode de 1987 a 2023. L'objectif principal est de developper des modeles de prevision fiables pour aider la prise de decision economique et financiere.

## Problematique

Le petrole Brent est l'un des benchmarks les plus importants pour les prix du petrole dans le monde. Prevoir son evolution est essentiel pour :
- Les decisions d'investissement dans le secteur energetique
- La planification budgetaire des entreprises dependantes du petrole
- Les politiques economiques des pays producteurs et consommateurs
- La gestion des risques financiers

## Dataset

Le jeu de donnees contient les prix journaliers du petrole Brent de mai 1987 a decembre 2023 :
- **Fichier** : `data/BrentOilPrices.csv`
- **Colonnes** : Date, Price (USD/baril)
- **Observations** : ~9 500 jours de trading
- **Plage de prix** : ~$8 - ~$196 USD

Le dataset est genere synthetiquement pour reproduire les caracteristiques reelles des prix du Brent, incluant :
- La hausse progressive liee a la demande chinoise (2002-2008)
- Le pic de 2008 et la crise financiere
- La stabilite a $100+ (2011-2014)
- La chute des prix OPEC (2014-2016)
- Le crash COVID-19 (2020)
- La hausse post-pandemie et le conflit ukrainien (2021-2022)

## Structure du Projet

```
brent-oil-forecasting/
├── data/
│   └── BrentOilPrices.csv          # Donnees des prix du Brent
├── figures/
│   ├── step1_visualization/         # Visualisations de base
│   ├── step2_decomposition/         # Decomposition de la serie
│   ├── step3_stationarity/          # Tests de stationnarite
│   ├── step4_transformations/       # Transformations
│   ├── step5_train_test_split/      # Division des donnees
│   ├── step6_naive_forecasting/     # Previsions naives
│   ├── step7_acf_pacf/              # Analyse ACF/PACF
│   ├── step8_statistical_models/    # Modeles statistiques
│   ├── step9_evaluation/            # Comparaison des modeles
│   ├── step10_autoarima/            # AutoARIMA
│   └── step11_ml_models/            # Modeles Machine Learning
├── notebooks/
│   └── brent_oil_forecasting.ipynb  # Notebook Jupyter complet
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Fonctions utilitaires
│   └── analysis.py                  # Pipeline d'analyse complet
├── generate_data.py                 # Generateur de donnees synthetiques
├── create_notebook.py               # Script de creation du notebook
├── requirements.txt                 # Dependances Python
└── README.md                        # Ce fichier
```

## Methodologie

### Etape 1 : Chargement et Visualisation
- Chargement du fichier CSV avec parsing des dates
- Visualisation de la serie temporelle complete
- Histogramme des moyennes annuelles
- Distribution des prix (histogramme + KDE)

### Etape 2 : Decomposition de la Serie Temporelle
- Reechantillonnage mensuel des donnees journalieres
- Decomposition additive (periode = 12 mois)
- Analyse des composantes : tendance, saisonnalite, residus

### Etape 3 : Tests de Stationnarite
- Statistiques glissantes (moyenne et ecart-type, fenetre = 252 jours)
- Test Augmented Dickey-Fuller (ADF)
- Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

### Etape 4 : Transformations
- Transformation logarithmique
- Differenciation de premier ordre
- Transformation Box-Cox
- Re-verification de la stationnarite apres chaque transformation

### Etape 5 : Division Train/Test
- Division chronologique 80% / 20%
- Visualisation de la coupure

### Etape 6 : Previsions Naives (Benchmarks)
- Last Value Naive
- Seasonal Naive (periode = 12 mois)
- Moyenne Mobile (12 mois)
- Lissage Exponentiel Simple
- Evaluation par MAE, RMSE, MAPE

### Etape 7 : Analyse ACF/PACF
- Autocorrelation et autocorrelation partielle de la serie originale
- ACF/PACF de la serie differenciee
- Identification des ordres potentiels AR/MA

### Etape 8 : Modeles Statistiques
- AR(2) : AutoRegressif d'ordre 2
- MA(2) : Moyenne Mobile d'ordre 2
- ARMA(2,2) : Combinaison AR + MA
- ARIMA(2,1,2) : ARMA avec differenciation integree
- SARIMA(1,1,1)(1,1,0,12) : ARIMA saisonnier

### Etape 9 : Evaluation et Comparaison
- Tableau comparatif de tous les modeles
- Metriques : MAE, RMSE, MAPE, AIC, BIC
- Graphiques de comparaison

### Etape 10 : AutoARIMA
- Selection automatique des parametres optimaux via pmdarima
- Prevision avec intervalles de confiance a 95%
- Comparaison avec le meilleur modele manuel

### Etape 11 : Modeles Machine Learning
- **Features** : retards (1, 7, 14, 30 jours), moyennes/ecarts-types glissants (7, 14, 30 jours), features calendaires
- **XGBoost** : Gradient Boosting avec regularisation
- **Random Forest** : Ensemble de forets aleatoires
- Previsions one-step-ahead
- Analyse de l'importance des features

## Resultats

| Modele | MAE | RMSE | MAPE |
|--------|-----|------|------|
| Moving Average (12) | 16.70 | 25.27 | 42.19% |
| ARMA(2,2) | 12.94 | 15.50 | 61.53% |
| AutoARIMA | 20.49 | 28.55 | 55.55% |
| XGBoost | 2.00 | 4.49 | 9.25% |
| Random Forest | 1.20 | 4.68 | 5.20% |

**Note** : Les modeles ML utilisent des previsions one-step-ahead avec acces aux prix passes comme features, ce qui n'est pas directement comparable aux previsions multi-step des modeles statistiques.

## Installation et Utilisation

### Prerequis
- Python 3.11+
- pip

### Installation

```bash
# Cloner le projet
cd brent-oil-forecasting

# Installer les dependances
pip install -r requirements.txt
```

### Generation des donnees

```bash
python generate_data.py
```

### Execution de l'analyse

```bash
python src/analysis.py
```

Cette commande execute les 11 etapes et genere tous les graphiques dans le dossier `figures/`.

### Notebook Jupyter

```bash
jupyter notebook notebooks/brent_oil_forecasting.ipynb
```

## Dependances

- **pandas** : Manipulation de donnees
- **numpy** : Calcul numerique
- **matplotlib** : Visualisation
- **seaborn** : Visualisation statistique
- **statsmodels** : Modeles de series temporelles
- **pmdarima** : AutoARIMA
- **scikit-learn** : Machine Learning
- **xgboost** : Gradient Boosting
- **scipy** : Fonctions scientifiques (Box-Cox)
- **nbformat** : Creation de notebooks

## Limites et Perspectives

### Limites
- Les donnees synthetiques, bien que realistes, ne capturent pas tous les evenements historiques
- Les modeles statistiques assument la linearite et la normalite des residus
- Les modeles ML en one-step-ahead ne sont pas directement comparables aux previsions multi-step

### Perspectives d'amelioration
- Integration de variables exogenes (production OPEC, PIB mondial, taux de change)
- Modeles GARCH pour la volatilite conditionnelle
- Reseaux de neurones recurrents (LSTM, GRU)
- Modeles hybrides combinant approches statistiques et ML
- Previsions multi-step pour les modeles ML

## Licence

MIT License
