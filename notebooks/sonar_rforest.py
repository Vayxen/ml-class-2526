import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", css_file="", auto_download=["ipynb"])

with app.setup:
    # Import completi
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Import parziali
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        recall_score,
        precision_score,
    )
    from random import randint


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset sonar con l'algoritmo della *random forest*
    """)
    return


@app.cell
def _():
    # Riga dei titoli
    titles = [f"Banda {number}" for number in range(0, 60)]
    titles.append("Oggetto")


    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        names=titles,
    )
    le = LabelEncoder()
    df["Oggetto"] = le.fit_transform(df["Oggetto"])

    # Scelta convenzionale in ML è di chiamare x la matrice dei dati
    # ed y il vettore delle classi etichettate

    x = df.iloc[:, :-1]
    y = df["Oggetto"].values.ravel()
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analisi (ed eventuale scaling) delle feature
    """)
    return


@app.cell
def _():
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train)
    # x_test_scaled = scaler.transform(x_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Applicazione del modello
    """)
    return


@app.cell
def _(x, y):
    seed = randint(0, 100)
    test_seed = 4211

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=seed, stratify=y
    )


    rf_classifier = RandomForestClassifier(
        n_estimators=501, max_samples=0.65, random_state=seed
    )

    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    return y_pred, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metriche di performance
    """)
    return


@app.cell
def _(y_pred, y_test):
    # Matrice di confusione
    confusion = confusion_matrix(y_test, y_pred)
    labels = ["Roccia", "Mina"]

    sns.heatmap(
        confusion, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Previsione")
    plt.ylabel("Etichetta reale")
    plt.show()

    # Metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)


    print(accuracy, precision, recall)
    return accuracy, precision, recall


@app.cell(hide_code=True)
def _(accuracy, mo, precision, recall):
    mo.md(rf"""
    Il modello ha mostrato le seguenti metriche di performance: 

    - Accuratezza: ${round(accuracy, 4) * 100}\%$
    - Precisione: ${round(precision, 4) * 100}\%$
    - Recall: ${round(recall, 4) * 100}\%$
    """)
    return


if __name__ == "__main__":
    app.run()
