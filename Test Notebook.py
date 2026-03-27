import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Jupyter notebook di Machine Learning
    !todo: diversi algoritmi, diversi notebook
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import sklearn as skl
    return


if __name__ == "__main__":
    app.run()
