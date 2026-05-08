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
    # from sklearn.model_selection import LeaveOneOut
    return


if __name__ == "__main__":
    app.run()
