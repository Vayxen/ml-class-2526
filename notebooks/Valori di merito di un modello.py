import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature scaling e performance di un modello

    Il bisogno di riscalare i valori associati alle feature deriva dal fatto che la base matematica dei modelli di apprendimento è algebra e statistica: dei valori sproporzionati (in eccesso o difetto) rispetto alla scala media del dataset può causare *gravi* errori di apprendimento che rendono il modello funzionalmente inutile.

    L'idea iniziale è quella di normalizzare i dati: la semplice normalizzazione della forma $x'_i = \dfrac{x_i}{\max \{x_i\}}$ riporta tutti i valori nell'intervallo $[0, 1]$, ma come menzionato sopra, in un dataset mal bilanciato nei suoi valori, ciò darà problemi al modello.
    Una prima soluzione allora è, nell'ipotesi di una distribuzione normale di dati, lo **standard scaling**:
    $$
    x'_i=(x_i - \mu)/\sigma
    $$
    Questo centra la media della distribuzione in 0 con una deviazione standard di 1.

    Un secondo metodo è la **normalizzazione min-max**:

    $$
    {\displaystyle x'_i={\frac {x_i-{\text{min}}\{x_i\}}{{\text{max}\{x_i\}}-{\text{min}\{x_i\}}}}}
    $$

    che dunque trasla l'intervallo generico delle misure da $[a,b]$ a $[-1,1]$.
    ***
    Il vantaggio dei due scaler sopra è la loro semplicità, ma **entrambi i metodi sono soggetti al problema degli *outlier***. Possiamo allora seguire due piste:
    - Se gli outlier sono in numero trascurabile rispetto al dataset potremmo decidere di scartarli in base al criterio di Chauvenet;
    - Se costituiscono un insieme non trascurabile delle misure dobbiamo scegliere un altro scaler.

    Una delle risposte al secondo punto la troviamo nel **robust scaling**:

    $$
    {\displaystyle x'={\frac {x-Q_{2}(x)}{Q_{3}(x)-Q_{1}(x)}}}
    $$
    dove $Q_n$ è *il valore del quartile n-esimo nell'insieme delle misure*. Questo metodo è particolarmente utile dunque in presenza di dati non molto "lineari" in proporzione.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Le metriche di performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    *Non tutti i modelli necessitano di dati scalati*. Alcuni ne sono poco influenzati (ad esempio modelli come il naive-Bayes e le foreste casuali), talvolta potrebbe essere anche dannoso se dei dati perdono relazioni di non linearità a causa dello scaling.
    """)
    return


if __name__ == "__main__":
    app.run()
