
# Predict ROI for USDT pairs on Binance

Ce script utilise des modèles de Machine Learning pour prédire le retour sur investissement (ROI) des paires de cryptomonnaies liées au USDT sur Binance. Il combine les prévisions de trois modèles différents : régression linéaire, réseaux de neurones LSTM et modèle ARIMA.



## Fonctionnement

Le script récupère les données historiques des paires USDT sur Binance en utilisant l'API de Binance. Il normalise les données, crée des jeux d'entraînement et de test, entraîne les modèles et effectue les prévisions. Enfin, il combine les prévisions des trois modèles pour obtenir une prédiction finale.

## Installation


Clonez le repo :

```bash
git clone https://github.com/heavnzor/binance_usdt_pairs_roi_prediction
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

Créez un fichier config.py contenant votre clé API Binance :

```bash
api_key = 'votre-clé-api'
```


## Utilisation

Le script peut être exécuté depuis la ligne de commande :

```bash
py prediction.py --period <période> --output_file <fichier_sortie>
```

<période> spécifie la période de temps pour laquelle prédire le ROI (par exemple, 1w, 2w, 1M, etc.).

<fichier_sortie> est le nom du fichier de sortie pour enregistrer les résultats. Ce paramètre est facultatif.

## Auteur
heavnz0r

## Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Vous voulez me payer un café ?
### Adresse XMR : 
475maQ4Dv976sqD1FhzE1iHzwAgCx5VhxZGQCHegAsyvaoJZehATmrMDhk8GUSQB5sWsid1pfVYHDXpXk6VRrnvG17xUuK3
