import marimo

__generated_with = "0.21.1"
app = marimo.App()

with app.setup:
    import polars as pl
    import numpy as np


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Primi concetti di selezione/riduzione delle feature

    Di notevole importanza, prima di dare un set di misure in pasto ad un algoritmo di apprendimento, è la selezione delle feature *realmente* di interesse così da aumentare la qualità del modello (e risparmiare anche spazio in memoria, problema fortemente rilevante con dataset sempre pià grandi). Si elencano ed elaborano di seguito alcune tecniche di interesse, con i loro vantaggi/svantaggi e casi d'uso.

    > Per **selezione** si intende l'eliminazione
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principal Component Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kernel-PCA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Discriminant Analysis

    ## LDA iterativa per la selezione
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
