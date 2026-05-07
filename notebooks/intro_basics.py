import marimo

__generated_with = "0.23.5"
app = marimo.App(
    width="medium",
    css_file="C:\\Users\\geome\\Documents\\GitHub\\ml-class-2526\\nord.css",
    auto_download=["ipynb"],
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Le basi di machine learning

    Il *machine learning* è una branca dello studio dell'intelligenza artificiale che si occupa di costruire algoritmi che, attraverso metodologie matematiche/algebriche, deduce e "impara" da un insieme di dati a partire da un insieme di caratteristiche e parametri forniti all'algoritmo, che siano misure, rilevamenti di un sensore, immagini, etc...

    # Principali tipologie di apprendimento

    Gli algoritmi di apprendimento e i paradigmi oggi utilizzati sono molteplici e in costante evoluzione/formulazione, ciascuno con velocità di convergenza sempre migliori (in base al compito svolto). Possiamo comunque ridurci a 3 grandi approcci (ciascuno dei quali ha eventualmente un suo sottotipo, ma non ci interessa per ora):
    - Apprendimento **supervisionato**: al computer vengono forniti dei dati di esempio e le etichette con cui devono essere correttamente identificati, con il fine ultimo di far sì che l'algoritmo deduca eventualmente come etichettare il set di dati in maniera autonoma
    - Apprendimento **non supervisionato**: a differenza del primo, **non vi sono etichette fornite**. Di fatto, il compito dell'algoritmo è quello di *trovare autonomamente un pattern nel dataset*
    - Apprendimento **per rinforzo**: l'algoritmo interagisce con un ambiente, venendo attivamente premiato o punito in base alle azioni che compie.

    **Nessuno dei 3 algoritmi** (così come i metodi matematici sfruttati in essi) **è *la* soluzione a tutti i problemi di apprendimento**, in quanto *dipende dal risultato desiderato dal modello di apprendimento e da ciò che si vuole fare con i dati*. Per fare un esempio per tutte le suddette categorie (ma di nuovo, senza scendere nel dettaglio dei diversi modelli matematici adottati),
    - l'apprendimento supervisionato è tendenzialmente usato *generalmente in qualsiasi analisi dov'è nota la caratteristica in esame e si vuole automatizzare questo processo di riconoscimento per dataset particolarmente grandi*. L'esempio semplice è quello di un insieme di immagini di animali pre-etichettati (quindi il set di dati *contiene già le risposte giuste*)
    - il non supervisionato serve a *studiare dataset nei quali si vogliano cercare caratteristiche comuni o raggruppare per caratteristica gruppi di dati* nei quali vi è qualche forma di correlazione statistica. Questo succede nei casi dove sono raccolte molte misure, ma non è possibile classificare in partenza cosa ogni dato possa rappresentare (compito che spetta all'algoritmo di apprendimento)
    - l'apprendimento per rinforzo è meglio utilizzato laddove l'algoritmo deve imparare in un ambiente che alla base è un *problema decisionale*: la mossa migliore in un gioco, pilotare autonomamente un veicolo mantenendolo in corsia e reagendo agli ostacoli, qualsiasi problema il cui procedimento si riassume con "trovare la via corretta/più efficiente che soddisfi la richiesta".


    # Le "features" e le "classi" di un set di dati

    In gergo, si chiamano *feature* i parametri associati ad un certo dato (se si pensa all'esempio dei dataset di immagini, potremmo considerare misure di lunghezza e larghezza ed altri valori geometrici, colore, ...). Chiamiamo invece *classi* le
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
