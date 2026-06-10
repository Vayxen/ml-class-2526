import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt


    from sklearn.svm import SVR

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Esempio di regressione attraverso la SVM
    """)
    return


@app.cell
def _(np, plt):
    x = np.linspace(0, 10)
    print(x)

    sine = np.sin(x)
    y = np.sin(x).ravel() + np.random.normal(0, 0.1)

    plt.scatter(x, y)
    plt.plot(x, sine)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
