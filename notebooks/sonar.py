import marimo

__generated_with = "0.21.1"
app = marimo.App(auto_download=["ipynb"])

with app.setup:
    import marimo as mo

    # Calcolo e plotting
    import pandas as pd
    import random
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Scikit-learn
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Dataset "sonar" e applicazione della LDA ad uno spazio di 60 feature

    Le 60 feature corrispondono alle misure dei sensori su 60 bande di frequenza. Il dataset è stato importato dal link fornito, appendendo (puramente come scelta di stile) una riga di titoli. Le classi (etichette) sono rappresentate dalle lettere R (Roccia) e M (Mina) in base all'oggetto rilevato dalle misure del sonar.

    Alla colonna delle etichette è stato applicato il `LabelEncoder` di `scikit-learn`, così da convertire le etichette, originariamente di tipo stringa, in valori numerici.

    La *linear discriminant analysis* è qui applicata come classificatore, scalando prima i dati numerici con lo standard scaling.
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

    # df["Oggetto"] = df["Oggetto"].replace({"R": 0, "M": 1}).infer_objects()

    # Quello sopra è un approccio manuale, tuttavia sklearn offre un metodo più diretto per occuparsi della codifica numerica di label di tipo letterale, come segue:
    le = LabelEncoder()
    df["Oggetto"] = le.fit_transform(df["Oggetto"])

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
    # per testare la performance del modello a partire da vari stati iniziali
    fix_seed = 42
    # per testare il funzionamento del codice

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
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
    y_pred = lda.fit(x_train_scaled, y_train).predict(x_test_scaled)
    # il dataset di test è come un foglio di verifica
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
    - Accuratezza: ${round(accuracy, 4) * 100}\%$
    - Precisione: ${round(precision, 4) * 100}\%$
    - Recall: ${round(recall, 4) * 100}\%$
    - F1-score: ${round(f1, 4) * 100}\%$
    """)
    return


if __name__ == "__main__":
    app.run()
