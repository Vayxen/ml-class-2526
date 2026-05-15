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
    # Partizionamento dei dati e ottimizzazione dei parametri

    Particolarmente importante è che i dati vengano partizionati nei set appropriati (training, test ed eventuale validation). Il validation è una specie di "pre-test" sul quale studiamo la performance del modello al variare dei parametri di apprendimento relativi al modello specifico utilizzato. Una volta arrivati al test set, stiamo testando il modelllo "definitivo".

    Per ottimizzare i parametri possiamo far ricorso a diversi approcci:
    - *brute force*, che prova tutte le possibili combinazioni
    - altro approccio non menzionato
    - altro ancora

    ## Approccio di forza bruta

    1. Il primo passaggio è definire lo spazio dei paramtri/iperparametri (ad esempio, negli alberi decisionali, esempi di iperparametri sono la profondità dell'albero, i campioni che ogni foglia deve contenere ad ogni suddivisione, etc..., o per il k-means il parametro è k...). Ad esempio, nel k-means, l'intervallo su cui facciamo variare k è generalmente $k \in \left[1, \dfrac{N_{classi} -1}{2}\right]$
    2. Si generano poi tutte le possibili combinazioni dei suddetti iperparametri. Chiaramente non lavoriamo su intervalli continui: quindi un intervallo della forma $[A_{\min},A_{\max}]$ avrà valori che decidiamo di equispaziare, ad esempio con $\Delta = \frac{A_{max}- A_{min}}{n}$. Questo implica ad esempio che con $N$ iperparametri che contengono $M$ valori possibili, le configurazioni totali sono $M^N$ (e ne deduciamo il costo computazionale elevato per modelli con tanti iperparametri).
    3. Si tentano tutte le sopracitate combinazioni, valutando la performance del modello in seguito (con le tecniche di valutazione della performance come ad esempio la cross-validation e quant'altro).
    4. Si seleziona dunque la combinazione che ha restituito la migliore performance.


    ## Tecniche di partizionamento

    ### Metodo *holdout*

    Il metodo finora utilizzato dove la proporzione training:test è predeterminata (valori comuni sono 80:20 o proporzioni vicine a questa, talvolta casi più estremi con un 50:50). Utile per modelli di *dimensioni medie o grandi*.

    ### Splitting stratificato

    Utile **quando i dataset sono sbilanciati**. Anche questo è stato finora utilizzato chiamando il parametro `stratify` nella funzione `train_test_split`: consiste nel mantenere la proporzione in fase di suddivisione. Se una classe ad esempio costituisce un terzo dei dati nel dataset, alla suddivisione, la funzione di sklearn fa sì che sia nel dataset training che quello test, questa classe in minoranza continui a costituire un terzo dei rispettivi insiemi.

    ### (k-fold) Cross-validation

    Serve principalmente a valutare la capacità di generalizzazione di un modello. Trattasi di una tecnica molto potente, poiché allena diversi modelli suddividendo più volte i dati in maniera diversa. Il metodo di cross validation più noto è il cosiddetto *k-fold*: consiste nel partizionare il dataset di $N$ elementi in $k$ partizioni di dimensioni uguali (o almeno, quasi uguali). Uno dei fold diventa il dataset test, mentre i fold rimanenti diventano il training. Itero questa procedura, cambiando quale dei $k$ fold è il dataset test.
    > $k$ di solito è 5 o 10.

    Il metodo è implementato attraverso i sottomoduli `cross_val_score, KFold`, dove il primo valuta la performance dei vari fold, mentre il secondo è quello che partiziona il dataset in $k$ parti (fold). Il metodo, chiaramente, per coprire tutte le possibili partizioni, esegue $k$ iterazioni. **La scelta di $k$ influenza di fatto lo split**: se $k=5$, la proporzione è $80:20$, se $k=10$, abbiamo $90:10$.
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import KFold  # , cross_val_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    model = LinearDiscriminantAnalysis()  # valori a caso per non avere errori
    x = [] * 10
    y = [] * 10
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # scores = cross_val_score(
    #  model, x, y, cv=kf
    # )  # cv può non essere passato, il valore di default è il 5-fold
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### *Leave-one-out* cross-validation

    Trattasi di un metodo particolarmente utile in dataset di dimensioni limitate, dove si adotta uno split di $N-1$ dati di training e di ***un unico*** elemento come test "set". Il modello deve dunque essere eseguito $N$ volte, ottenendo $N$ classificatori ed $N$ esiti, valutando la prestazione del modello attraverso la media dei $N$ risultati. Utilizza al massimo il dataset, ma chiaramente è computazionalmente costoso, poiché significa eseguire un modello di apprendimento più o meno complicato $N$ volte. Per questo motivo è pratico ove il dataset contenga "pochi" $N$ dati (sulle decine/centinaia).
    """)
    return


@app.cell
def _():
    from sklearn.model_selection import LeaveOneOut

    loocv = LeaveOneOut()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Trattandosi di un metodo di partizionamento particolarmente/estremamente "esaustivo", vanno fatte alcune considerazioni sul campo di applicazione. Sia ad esempio un dataset di 1000 vettori di dati. Con il L1O, se nel dataset è presente un gruppo di dati particolarmente correlati, l'insieme di classificatori nei quali il singolo rimosso è ciascuno dei dati correlati tra loro finirà per non mostrare performance/metriche particolarmente diverse. Di fatto, pur avendo rimosso il dato "problematico", i dati correlati all'interno del training set causano qualche forma di bias nel modello finale. Per ovviare a questo problema, invece di rimuovere necessariamente il singolo, possiamo decidere di rimuovere un **gruppo** di dati
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time-series splitting

    Come suggerisce il nome, è utile nei dataset dove è presente una dipendenza temporale (ad esempio misure prolungate nel tempo, dunque mercato finanziario/meteo/qualsiasi sensore che misura continuamente un dato fenomeno). Lo splitting è allora eseguito in modo tale che il training faccia da "storico" delle misure, laddove il dataset di test è costituito di dati tutti temporalmente successivi al training (come se fossero delle rilevazioni successive). Permette, concettualmente, che il modello effettui previsioni sull'andamento delle misure nel tempo, simulando anche il possibile andamento per dati non ancora ricevuti. Pandas è molto comodo su questo fronte: possiamo effettuare un sorting di tutta la matrice dei dati sfruttando la colonna che rappresenta l'istante di rilevazione. Effettuata la suddivisione temporale, il metodo di partizione `TimeSeriesSplit` può sfruttare ulteriori tecniche di partizione come il k-fold in fase di addestramento del modello.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data augmentation

    Siano ad esempio dei dati ottenuti da un esperimento: può capitare di dover sintetizzare dei dati attraverso interpolazione (simulando ulteriori dati in base alla distribuzione dei dati reali). I motivi di questa scelta potrebbero derivare ad esempio da un forte sbilanciamento dei dati, per motivi di natura strumentale o proprio fisica (perché è possibile che l'evento in esame sia particolarmente raro). L'idea non è solo fisica: potremmo incrementare la risoluzione di un file audio o immagini (cosa che accade ad esempio con le reti neurali convoluzionali). Esistono vari metodi e tecniche: si elencano alcuni esempi.

    ## SMOTE (*Synthetic Minority Oversampling TEchnique*)

    Come suggerisce il nome, il metodo genera dati aggiuntivi specificamente della classe in difetto, al fine di bilanciare meglio il dataset. Il funzionamento è il seguente:
    1. Si seleziona un'istanza della classe minoritaria
    2. Si effettua un k-nearest neighbors (solitamente k=5)
    3. Si seleziona uno dei vicini casualmente
    4. Si interpola tra il dato selezionato di riferimento ed il vicino, considerando (geometricamente parlando) un punto sul segmento che li collega, con un parametro di distanza casuale tra 0 ed 1
    5. Si ripete questo processo quante volte si ritiene necessario.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Un'applicazione dei metodi sopracitati

    Lo si applica al database *breast*, in particolare facciamo un esempio con la 10-fold CV ed il modello Support Vector.
    """)
    return


@app.cell
def _():
    # Import parziali

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix

    # Import totali

    import pandas as np
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assegnazione del dataset completo, delle misure e della colonna delle etichette
    dataset = load_breast_cancer
    x = dataset.data
    y = dataset.target

    # Standardizzo

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Carico il modello e sfrutto la 10-fold cross-validation

    model = SVC(C=3.0)
    # di default usa kernel rbf e gamma = "scale"
    # inoltre C è il parametro che fa da "costo" sugli errori
    cv_scores = cross_val_score(model, x_scaled, y=y, cv=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Così facendo stiamo addestrando il modello su 10 partizionamenti diversi dei dati (come addestrare 10 modelli diversi).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TODO: continualo guardando il file ipynb
    """)
    return


if __name__ == "__main__":
    app.run()
