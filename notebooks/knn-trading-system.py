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
    # Applicazione del kNN al trading

    ## Definizioni iniziali
    Definisco in primis un sistema di trading: è un insieme di regole e procedure utilizzate per automatizzare le azioni di acquisto e vendita di uno strumento finanziatio, sulla base dell'analisi dell'andamento del mercato e delle previsioni di un modello di ML.

    (Per ulteriori concetti di base, ["Introduzione all'analisi tecnica"](http://eprints.biblio.unitn.it/303))

    La cronologia dei dati di un certo prodotto/azione/etc. sono importanti per i seguenti motivi

    1. Sviluppo e valutazione di strategie
    2. Analisi dell'andamento del mercato
    3. Studio e ottimizzazione dei parametri ("di fit", per così dire) relativi al sistema
    4. Adattamento a dati futuri/comportamenti mutevoli

    Dati di interesse nei suddetti dataset sono prezzo, volume (di azioni scambiate per un titolo in un certo $\Delta t$), indicatori/parametri di carattere più matematico e risposta/variazione di un andamento finanziario relativamente ad un evento di attualità (*crisi dei semiconduttori* ad esempio).

    Il prezzo di un'azione è determinato, come molti beni concreti e non, dalla curva di domanda e offerta, le quali sono influenzate principalmente da fattori vari (quali la performance dell'azienda come fattore più "interno", il sentimento del mercato o eventi esterni che li influenzano come fattori esterni).

    Definisco 3 termini tecnici:
    - l'*utile* è definito come la differenza tra il guadagno dell'azienda e il suo costo di operazione. Se positiva è in guadagno, se negativa è in perdita o *deficit*.
    - i *dividendi* sono quella parte di utile che viene data dall'azienda agli azionisti (una sorta di stipendio/premio per gli investitori)
    - si dice *gain* (guadagno o perdita che sia)

    Gli investitori adottano strategie in base al loro obiettivo (un guadagno veloce o un portafoglio alla lunga): tra questi fattori che decidono il da farsi di un investitore figurano
    - l'orizzonte temporale
    - il rischio che intendono correre
    - le condizioni di mercato
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementazione del modello

    Il modello qui presentato effettua una (presumo rudimentale, l'esempio è didattico) analisi finanziaria di un certo mercato (l'esempio del professore è Tesla, ma vorrei farne un altro) importando i dati dalla libreria di Yahoo Finance (`yfinance`) in un intervallo di tempo specifico. Essendo la libreria particolarmente limitante nelle richieste (essendo svolte solo online) si potrebbe invece importare qualche copia disponibile in qualsiasi momento, ma si farà un primo esempio con la libreria.
    """)
    return


@app.cell
def _():
    import yfinance as yf
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    return


if __name__ == "__main__":
    app.run()
