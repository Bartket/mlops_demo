# MLOps demo with Dagster and MLFlow

## Description
Wine Quality prediction model based on Wine dataset

https://archive.ics.uci.edu/ml/datasets/wine

Data Set Information:

These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

I think that the initial data set had around 30 variables, but for some reason I only have the 13 dimensional version. I had a list of what the 30 or so variables were, but a.) I lost it, and b.), I would not know which 13 variables are included in the set.

The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it )

1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline
## Configure environment (pyenv based on Python 3.8.10)
```console
pyenv virtualenv 3.8.10 mlops_demo
pyenv local mlops_demo
pyenv shell mlops_demo
pip install -r requirements.txt
```

## To start run

```console
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow ui \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0

dagit -f src/main.py
```