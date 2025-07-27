# Black-Scholes Option Pricing

Ce projet implémente le modèle de Black-Scholes pour le pricing d'options européennes. Il inclut une explication théorique, une implémentation en Python, une simulation Monte Carlo, ainsi qu'une visualisation de la sensibilité des prix (Greeks).

## Contenu

- Implémentation de la formule Black-Scholes pour options call/put
- Simulation de trajectoires de prix par mouvement brownien géométrique
- Estimation du prix d'une option par Monte Carlo
- Visualisation des "Greeks" : delta, gamma, vega, theta, rho

## Fichiers

- `bs_pricing.ipynb` : notebook principal
- `src/black_scholes.py` : fonctions utiles (pricing, greeks, simulations)
- `requirements.txt` : dépendances

## Exécution

```bash
pip install -r requirements.txt
jupyter notebook bs_pricing.ipynb
