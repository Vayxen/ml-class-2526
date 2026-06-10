import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metodi di *ensemble learning*

    Si dicono tali quei metodi che fanno uso di più modelli di apprendimento al fine di migliorare la prestazione complessiva e la robustezza di un sistema di previsione. Il motivo è che la diversità di metodi di apprendimento può effettivamente portare a capacità di generalizzazione migliori del singolo modello (in un modo concettualmente reminiscente della *saggezza della folla*). I 3 ensemble più comuni sono gli ensemble ...
    1. ... basati sull'apprendimento
    2. ... basati sulle caratteristiche
    3. ... ibridi

    ## Basati sull'apprendimento

    Esempi di questi metodi sono:

    - ***bagging*** (slang per *bootstrap aggregating*), che crea diversi insiemi di addestramento, utilizzati su modelli diversi, combinando il risultato finale per media o maggioranza, come le foreste casuali; questo riduce la varianza delle previsioni effettuate dai singoli modelli, guadagnando dunque in stabilità;
    - ***boosting***, dove l'addestramento avviene in maniera sequenziale (ogni addestramento successivo avviene basandosi sulle performance precedenti, come l'*XGBoost* o *Adaboost*): i modelli imparano in particolare assegnando dei pesi maggiori agli errori dei modelli precedenti. Trova largo utilizzo in apprendimento supervisionato, sia nella classificazione di dati etichettati che nella regressione;
    - ***stacking***, che addestra un *meta-*modello al fine di renderlo un "classificatore di classificatori", con l'obiettivo di apprendere *come combinare le previsioni dei modelli di base*.


    ## Basati sulle caratteristiche

    - Random Subspace Method
    - Bootstrap Aggregated Feature Selection


    ## Configurazioni ibride

    - Boosting + selezione delle caratteristiche
    - Stacking + trasformazione delle caratteristiche

    ## Come vengono combinati i risultati nei metodi di ensemble

    - Maggioranza: si suppone che ogni classificatore dia la sua previsione di un certo dato di test, allora il risultato finale è la previsione più popolare
    - Media: laddove la previsione sia di natura numerica, la previsione finale è semplicemente la loro media. Risulta utile in problemi di regressione/interpolazione.
    - Media pesata: come sopra, ma con dei pesi assegnati ai modelli in base a quanto questi siano accurati/generalmente affidabili
    - Stima delle probabilità:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Esempio di applicazione pratica con il database breast
    """)
    return


@app.cell
def _():
    # Import parziali

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

    from sklearn.ensemble import VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    # Import completi

    import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    L'idea qui sarà di applicare il voting di 3 modelli (kNN, albero decisionale e SVM) e mostrare come i 3 modelli, combinati, mostrano effettivamente una capacità di previsione migliore dei singoli modelli.
    """)
    return


@app.cell
def _():
    pass
    return


if __name__ == "__main__":
    app.run()
