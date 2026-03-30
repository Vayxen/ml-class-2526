import marimo

__generated_with = "0.21.1"
app = marimo.App(auto_download=["ipynb"])

with app.setup:
    import marimo as mo
    import pandas as pd

    # import numpy as np
    import random

    # import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler  # , LabelEncoder

    # from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split


@app.cell
def _():
    titles = [f"Banda {number}" for number in range(0, 60)]
    titles.append("Oggetto")


    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        names=titles,
    )


    # Roccia = 0, Mina = 1

    df["Oggetto"] = df["Oggetto"].replace({"R": 0, "M": 1})

    x = df.iloc[:, :-1]
    y = df["Oggetto"].values.ravel()

    df.head()
    return x, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Scaling delle feature e applicazione della LDA
    """)
    return


@app.cell
def _(x, y):
    seed = random.randint(0, 50)
    fix_seed = 42

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=fix_seed, stratify=y
    )

    lda = LinearDiscriminantAnalysis()
    sc = StandardScaler()

    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    return lda, x_test_scaled, x_train_scaled, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Esecuzione del modello
    """)
    return


@app.cell
def _(lda, x_test_scaled, x_train_scaled, y_train):
    y_pred = lda.fit(x_train_scaled, y_train).predict(
        x_test_scaled
    )  # il dataset di test è come un foglio di verifica
    return (y_pred,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Metriche di performance
    """)
    return


@app.cell
def _(y_pred, y_test):
    from sklearn.metrics import confusion_matrix

    confmat = confusion_matrix(y_test, y_pred)
    labels = ["Roccia", "Mina"]

    sns.heatmap(
        data=confmat,
        annot=True,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Previsione")
    plt.ylabel("Risposta reale")
    plt.show()
    return


@app.cell(hide_code=True)
def _(y_pred, y_test):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    # Calcolo delle figure di merito
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Stampa delle figure di merito
    return accuracy, f1, precision, recall


@app.cell(hide_code=True)
def _(accuracy, f1, precision, recall):
    mo.md(rf"""
    Il modello appena eseguito ha restituito le seguenti metriche: 
    - Accuratezza: ${round(accuracy, 3) * 100}\%$
    - Precisione: ${round(precision, 3) * 100}\%$
    - Recall: ${round(recall, 3) * 100}\%$
    - F1-score: ${round(f1, 3) * 100}\%$
    """)
    return


if __name__ == "__main__":
    app.run()
