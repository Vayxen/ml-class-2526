import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    # Import completi
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    # import seaborn as sns


    # from-import
    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Applicazione del k-means: database dei vini
    """)
    return


@app.cell
def _():
    wine_set = load_wine()
    return


if __name__ == "__main__":
    app.run()
