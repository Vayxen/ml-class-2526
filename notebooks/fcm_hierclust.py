import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fuzzy C-Means (FCM)

    Anche detto algoritmo *soft k-means*, si distingue dal modello originale poiché non assegna un elemento ad un cluster *con certezza*, ma tiene conto di una probabilità associata a ciascun punto di appartenere al cluster (quantificata da un valore nell'intervallo $[0,1$]). In particolare, **un punto si può dire appartenere a più cluster, insieme ad un certo grado di appartenenza su ciascuno di essi**.

    Visualmente, i cluster di dati non sono allora nettamente separati, ma vedono una transizione più graduale/si osservano cluster con zone di sovrapposizione: un esempio possono essere animali che appartengono ad una stessa "famiglia", considerando gli eventuali ibridi o sottotipi che non possono essere totalmente piazzati in *una* famiglia o l'altra. Chiaramente, se un punto appartiene a più cluster, la somma dei *gradi di appartenenza* deve essere $1$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering gerarchico

    Il clustering gerarchico è un algoritmo di clustering che raggruppa degli elementi e li mette in relazione per similarità, raggruppandoli successivamente in cluster più ampi possibile. Nello specifico, questo è detto clustering ***agglomerativo***.
    Si ha anche il clustering **divisivo**, dove si parte da grandi cluster di dati e si suddivide iterativamente, anche fino a ridurli all'unità.

    In entrambi i casi, i cluster sono visualizzabili attraverso i *dendrogrammi*, che evidenziano relazioni tra coppie di elementi e, salendo, tra coppie di cluster.

    ## Utilizzo in machine learning

    Un esempio è identificare non singoli clienti di un negozio (ad esempio supermercati), ma *gruppi di clienti per caratteristiche omogenee*

    **Pro**:
    - (forse semplice implementazione?)

    **Contro**:
    - Costo computazionale alto (di base ha complessità temporale cubica e richiede un quantitativo circa quadratico di memoria)
    """)
    return


if __name__ == "__main__":
    app.run()
