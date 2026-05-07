import marimo

__generated_with = "0.23.5"
app = marimo.App()

with app.setup:
    # import polars as pl
    import numpy as np


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Primi concetti di selezione/riduzione delle feature

    Una volta applicato un dato modello di apprendimento, se nel dataset di addestramento i parametri di merito sono alti ma *eccessivamente alti rispetto alla performance sul dataset di test*, questo è un possibile indicatore di *overfitting* (il modello si è adattato al dataset di addestramento, ma non generalizza con successo ulteriori dati). Mitigare l'*overfitting* è possibile in vari modi, tra cui

    - aumentare i dati di addestramento
    - considerare un modello più semplice
    - **ridurre la dimensionalità dei dati**

    L'ultimo metodo è altresì noto come *feature selection*.
    Di notevole importanza, prima di dare un set di misure in pasto ad un algoritmo di apprendimento, è la selezione delle feature *realmente* contribuenti all'apprendimento, così da aumentare la qualità del modello (e risparmiare anche spazio in memoria, riducendo la complessità spaziale e temporale di molto). Si elencano ed elaborano di seguito alcune tecniche, con i loro vantaggi/svantaggi e casi d'uso.

    Esistono due diciture: **riduzione** e **selezione** delle feature (o più nello specifico si parla di riduzione della dimensionalità o selezione delle feature). La differenza concettuale è che la selezione delle feature considera lo spazio $\mathbb{R}^n$ e seleziona $k$ feature. Le restanti $n-k$ __non sono distrutte__: semplicemente non partecipano al calcolo. Questo può essere utile laddove il modello migliora con meno feature, ma le feature restanti sono di interesse statistico nell'analisi dei dati e rimangono comunque associabili direttamente ai dati analizzati (ad esempio, potremmo scartare l'età in un algoritmo che cerca di fare distinzioni tra animali in base alle loro caratteristiche fisiche, ma potremmo voler mantenerla poiché di interesse a fine statistico). La riduzione invece **trasforma la base dello spazio di partenza** in una base di dimensione $k<n$, le cui componenti sono combinazioni lineari delle componenti originali. Di fatto, nell'analisi dei dati, i parametri non sono più singole caratteristiche (ad esempio, nel caso del dataset dei vini, contenuto alcolico-flavonoidi-fenoli-etc...), ma caratteristiche combinate (e moltiplicate per qualche coefficiente) che fungono da singolo parametro.

    A questo scopo possiamo identificare (tra molte possibili scelte) due metodi principali, che hanno applicazione generale in analisi dati e trovano di fatto utilizzo in ML.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Principal Component Analysis

    > si omettono, forse temporaneamente o forse no, i dettagli completi matematici, qui non di particolare interesse pratico

    La PCA è un metodo di **riduzione** (per enfasi: non è di selezione) che da un punto di vista teorico *considera lo spazio originale delle feature ed identifica una nuova base di vettori, lungo i quali si ha la massima varianza*.

    Da un punto di vista visuale/algebrico, stiamo considerando un sottospazio $k$-dimensionale sul quale proiettiamo i dati: potremmo ad esempio considerare una nuvola di dati tridimensionale e considerare il piano lungo le cui direzioni la varianza dei dati è maggiore. La PCA è conveniente **quando i dataset esprimono effettivamente delle varianze particolarmente alte lungo specifiche direzioni**: se la distribuzione dei dati nello spazio originale è pressoché sferica, tutte le feature sono relativamente importanti e discriminanti ed isolare delle componenti principali non è possibile (se non addirittura fortemente dannoso, poiché rimuoviamo di fatto possibilità di discriminazione tra classi).

    **Svantaggi della PCA**:
    - essendo un metodo che funziona di base attraverso trasformazioni lineari, ignora eventuali caratteristiche/relazioni di non linearità presenti nei dati;
    - massimizzando la varianza attraverso una proiezione su un sottospazio vettoriale, non tiene conto della capacità di discriminazione delle feature: ciò implica che se nello spazio originale ci sono degli insiemi di feature altamente discriminanti (e che quindi normalmente identificano con facilità e accuratezza la classe a cui appartengono quei dati), si va a perdere questa capacità di discriminazione sui nuovi vettori di feature;
    - è puramente un algoritmo di analisi dati e **non funge da classificatore**: la PCA fa da step di preprocessing dei dati, ed il suo output viene dato in pasto ad un algoritmo di apprendimento (knn, alberi decisionali, etc...).

    > Questo si vede anche nel pratico perché nelle librerie di `sklearn`, seppur sia presente la PCA, non ha un metodo `.predict()` associato.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Kernel-PCA

    Il metodo *kernel* PCA è, come la PCA, una tecnica di riduzione della dimensionalità dei dati, in particolare **proietta i dati su uno spazio di dimensione superiore con lo scopo di aumentarne la separabilità** (e da lì identificare le componenti di massima varianza). (espandi?)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Discriminant Analysis
    Per LDA si intende un metodo di riduzione della dimensionalità (**che può talvolta essere utilizzato direttamente come classificatore**) che proietta i dati lungo degli assi nuovi **tale che le proiezioni sugli assi nuovi mostrano forte separazione** (ed è per questo motivo che può anche fungere da classificatore). Di base, la LDA fa uso del criterio di Fisher (il criterio che stabilisce quantitativamente la sovrapposizione/capacità di discriminazione di una coppia di feature). Implicitamente si formano allora dei cluster di dati ben raggruppati (alta similarità intraclasse) e ben separati (bassa similarità interclasse). Mentre nella PCA la scelta delle PC volute dipende dal criterio "quante componenti conservano il 90-95% della varianza nei dati", nella LDA il massimo numero di feature è dato da
    $$
    \min(\text{numero di feature},\text{numero di classi}-1)
    $$
    Ad esempio, in un dataset di 13 feature e 3 classi (quello del vino), la dimensione del dataset trasformato è 2. Questo non è necessariamente un problema: è possibile che avere "così pochi parametri" rende anzi l'algoritmo capace di stabilire più facilmente (intendesi: con meno calcoli) quando un certo punto è oltre dei valori di soglia

    **Svantaggi**:
    - il metodo è utilizzabile solo in apprendimento supervisionato, le feature su cui opera la separazione devono essere note in partenza
    - soffre per costruzione di svantaggi simili alla PCA riguardo la linearità che trascura eventuali nonlinearità e la perdita di informazioni a causa della proiezione geometrica

    ## LDA iterativa per la selezione

    Alla riduzione della dimensionalità operata dalla LDA si aggiunge un'eventuale procedimento di selezione delle feature. Passo-passo,

    1. Si parte con il vettore/la lista di feature iniziali inalterata, a partire dal quale si applica la LDA.
    2.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
