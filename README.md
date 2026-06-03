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
│   ├── step11_ml_models/            # Modeles Machine Learning
│   ├── step12_model_analysis/       # Analyse des modeles
│   └── step13_final_comparison/     # Comparaison finale V1 vs V2
├── notebooks/
│   └── brent_oil_forecasting.ipynb  # Notebook Jupyter complet
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Fonctions utilitaires
│   ├── analysis.py                  # Pipeline d'analyse complet (V1)
│   ├── analysis_v2.py              # Analyse corrigee (V2 - comparaison equitable)
│   └── final_comparison.py         # Comparaison finale et figures recapitulatives
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

**Note** : Les resultats ci-dessus proviennent de l'analyse V1 ou les modeles ML utilisent des previsions one-step-ahead. Voir la section "Analyse Corrigee (V2)" ci-dessous pour une comparaison equitable.

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

## Analyse Corrigee (V2)

Le fichier `src/analysis_v2.py` corrige trois problemes critiques identifies dans l'analyse originale :

### Correction 1 : Comparaison equitable (multi-step vs one-step)

**Probleme** : Les modeles ML (XGBoost, Random Forest) utilisaient la prediction one-step-ahead avec les valeurs reelles en lag, tandis que les modeles statistiques (ARIMA, SARIMA) faisaient de la prevision multi-step. Cela rendait la comparaison injuste.

**Solution** : Implementation de la prevision recursive multi-step pour les modeles ML :
- Le modele predit t+1
- Cette prediction devient lag pour predire t+2
- Continue recursivement pour tout l'horizon de test

### Correction 2 : Dominance du lag_1

**Probleme** : La feature lag_1 faisait que les modeles copiaient essentiellement le prix de la veille, cachant s'ils apprenaient de vrais patterns.

**Solution** :
- Suppression de lag_1 des features
- Utilisation des lags 7, 14, 30 uniquement
- Ajout de features log-return : `log(prix_t) - log(prix_t-k)` pour k = 7, 14, 30

### Correction 3 : Choix additif vs multiplicatif

**Probleme** : La decomposition additive etait codee en dur sans justification.

**Solution** : Verification automatique :
- Division de la serie en 4 segments egaux
- Calcul de l'ecart-type des residus dans chaque segment
- Si correlation(niveau, ecart-type) > 0.5 : multiplicatif
- Sinon : additif
- Le choix est documente et justifie dans la figure generee

### Execution de l'analyse corrigee

```bash
python src/analysis_v2.py
```

### Resultats corrigees (comparaison equitable)

| Modele | MAE | RMSE | MAPE (%) |
|--------|-----|------|----------|
| ARIMA(2,1,2) | 20.43 | 28.50 | 55.30 |
| AutoARIMA | 20.49 | 28.55 | 55.55 |
| XGBoost (Recursif) | 30.82 | 34.88 | 177.94 |
| Random Forest (Recursif) | 40.88 | 49.08 | 224.98 |
| SARIMA(1,1,1)(1,1,0,12) | 59.09 | 70.62 | 195.03 |

**Conclusion** : Avec une comparaison equitable en multi-step, ARIMA surpasse les modeles ML. La superiorite apparente des modeles ML dans V1 etait due a l'utilisation de lag_1 (copie du prix precedent) et de la prediction one-step-ahead.

## Limites et Perspectives

### Limites
- Les donnees synthetiques, bien que realistes, ne capturent pas tous les evenements historiques
- Les modeles statistiques assument la linearite et la normalite des residus
- En prevision multi-step recursive, les erreurs ML s'accumulent rapidement

### Perspectives d'amelioration
- Integration de variables exogenes (production OPEC, PIB mondial, taux de change)
- Modeles GARCH pour la volatilite conditionnelle
- Reseaux de neurones recurrents (LSTM, GRU)
- Modeles hybrides combinant approches statistiques et ML
- Techniques de reduction d'erreur pour la prevision recursive (bootstrapping, etc.)

## Resultats et Evolution du Projet

### V1 -- Resultats initiaux (biaises)

Les premiers resultats semblaient montrer une superiorite des modeles ML :
- Random Forest : MAPE 5.20%
- XGBoost : MAPE 9.25%
- ARIMA(2,1,2) : MAPE 61.53%

**Mais ces resultats etaient trompeurs** pour trois raisons :

### Problemes identifies

#### 1. Comparaison deloyale (one-step vs multi-step)
Les modeles ML utilisaient la prediction one-step-ahead (avec le prix reel de la veille comme input), tandis que les modeles statistiques faisaient du multi-step forecasting sur tout l'horizon de test. C'est comme comparer un etudiant qui a les reponses avec un qui passe l'examen normalement.

#### 2. lag_1 qui ecrase tout
La feature lag_1 (prix de la veille) representait >80% de l'importance. Les modeles ne faisaient que copier le prix precedent -- ce n'est pas de la prevision, c'est de la persistance.

#### 3. Overfitting massif
Avec max_depth=15 sur seulement 260 observations mensuelles, Random Forest memorisait chaque point d'entrainement (score=1.0) sans generaliser (validation oscillant entre -0.4 et 0.7).

### V2 -- Resultats corriges (equitables)

Apres correction de ces trois problemes :

| Modele | MAE (USD) | Statut |
|--------|-----------|--------|
| ARIMA(2,1,2) | 20.43 | Meilleur |
| AutoARIMA(2,1,0) | 20.49 | Tres proche |
| SARIMA(1,1,1)(1,1,0,12) | 28.13 | Diverge |
| Random Forest (recursif) | ~73 | Erreur accumulee |
| XGBoost (recursif) | ~75 | Erreur accumulee |

### Conclusion academique

> **Sur une serie financiere univariee avec ~260 observations mensuelles et peu de features exogenes, les modeles statistiques (ARIMA) surpassent significativement les modeles ML en prevision multi-step.**

Les raisons :
1. Les arbres de decision ne peuvent pas extrapoler au-dela du range observe pendant l'entrainement
2. L'erreur recursive s'accumule exponentiellement sur un horizon de 85 mois
3. ARIMA est structurellement concu pour les series temporelles univariees
4. Avec peu de donnees, la parcimonie des modeles statistiques est un avantage

### Figures detaillees

Voir le dossier `figures/step13_final_comparison/` pour les visualisations completes de cette analyse comparative.

## Licence

MIT License
