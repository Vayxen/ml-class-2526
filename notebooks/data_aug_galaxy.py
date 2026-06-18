import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In questo esempio si sfrutta il *data augmentation* per aumentare la dimensione del set di addestramento, essendo il dataset (naturalmente) carente. Si importa un'immagine della galassia NGC3509, in scala di grigi per ridurre la dimensione dell'input.

    L'immagine è preprocessata, normalizzando rispetto al massimo valore della scala di grigi (valore compreso in $[0,255]$)


    (utilizza `np.where` per rimpiazzare condizionalmente pixel dell'immagine, oppure altera l'immagine in altri modi quali rescaling o rotazione, il motivo delle trasformazioni è addestrare la rete alla generalizzazione, cercando di renderla "robusta" rispetto a qualche forma di rumore nei dati)
    """)
    return


if __name__ == "__main__":
    app.run()
