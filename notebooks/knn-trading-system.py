import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Applicazione del kNN al trading

    Definisco in primis un sistema di trading: è un insieme di regole e procedure utilizzate per automatizzare le azioni di acquisto e vendita di uno strumento finanziatio, sulla base dell'analisi dell'andamento del mercato e delle previsioni di un modello di ML.

    (Per ulteriori concetti di base, ["Introduzione all'analisi tecnica"](http://eprints.biblio.unitn.it/303))

    La cronologia dei dati di un certo prodotto/azione/etc. sono importanti per i seguenti motivi

    1. Sviluppo e valutazione di strategie
    2. Analisi dell'andamento del mercato
    3. Studio e ottimizzazione dei parametri ("di fit", per così dire) relativi al sistema
    4. Adattamento a dati futuri/comportamenti mutevoli

    Dati di interesse nei suddetti dataset sono prezzo, volume (di azioni scambiate per un titolo in un certo $\Delta t$), indicatori/parametri di carattere più matematico e risposta/variazione di un andamento finanziario relativamente ad un evento di attualità (*crisi dei semiconduttori* ad esempio)
    """)
    return


if __name__ == "__main__":
    app.run()
