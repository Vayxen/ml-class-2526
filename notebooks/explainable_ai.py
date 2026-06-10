import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # L'intelligenza artificiale *spiegabile*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dicesi "IA spiegabile" quella particolare branca del machine learning/studio dei modelli di IA con lo scopo di rendere più trasparente il processo di apprendimento, specie delle reti neurali profonde. L'idea giace nel fatto che è spesso (se non sempre) sconosciuto quello che avviene negli strati nascosti di una rete neurale, non è cioè chiaro il meccanismo con cui una rete neurale trova dei pattern o delle feature discriminanti su cui basa il suo output finale. Esistono dunque dei modelli con il preciso scopo di ridurre questa astrazione ed evidenziare le suddette feature utilizzate
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Esempi: Occlusion e Grad-CAM
    """)
    return


if __name__ == "__main__":
    app.run()
