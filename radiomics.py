import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    # import parziali incompleti poiché usano tensorflow, che non riesco a far girare. Gira il file risultante su marimo online.
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout
    # from tensorflow.keras.optimizers
    # from tensorflow.keras.callbacks
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Radiomica
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La radiomica è una particolare branca della medicina dedicata all'applicazione del *machine learning* alla medicina (e in particolare al riconoscimento di immagini/analisi mediche attraverso algoritmi di apprendimento) con il fine di assistere nell'analisi, diagnosi e decisione in merito ad eventuali patologie.

    Un esempio pratico si può mostrare con l'applicazione delle CNN ad un dataset di immagini del torace, con il fine di rilevare casi normali e casi di polmonite. Il database contiene 5856 immagini, suddivise in train-test-validation (reperibile da [qui](https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images)). I dati sono volutamente sbilanciati verso i casi positivi di polmonite (in rapporto circa 1:3), così da permettere un miglior apprendimento dei segnali correlati alla presenza della patologia.
    """)
    return


if __name__ == "__main__":
    app.run()
