import marimo

__generated_with = "0.23.8"
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

    Il *machine learning* è una branca dello studio dell'intelligenza artificiale che si occupa di costruire algoritmi che, attraverso metodologie matematiche/algebriche, deduce e "impara" da un insieme di dati a partire da un insieme di caratteristiche e parametri forniti all'algoritmo, che siano misure, rilevamenti di un sensore, immagini, etc... .
    Nel pratico, i suoi campi di utilizzo sono innumerevoli: può automatizzare la ricerca di particolari rilevamenti in enormi insiemi di misure come quelle sopracitate, può effettuare delle stime/previsioni di carattere statistico su delle serie temporali, o può imparare ad eseguire compiti di ottimizzazione/di carattere decisionale. Il nome è particolarmente generico, in quanto vi sono moltissime sottobranche del machine learning, ciascuna con un obiettivo ben preciso, che siano gli esempi forniti sopra, o la sua applicazione alla robotica, o alla medicina per eventuali apparecchi di analisi/previsione di rischi alla salute, o la traduzione tra lingue, chi più ne ha più ne metta. Quella che oggi conosciamo come *intelligenza artificiale* altro non è che un'applicazione del machine learning, in quanto si tratta di sfruttare i modelli di analisi dei dati per simulare capacità deduttive/logiche sui dati ricevuti.

    # Principali tipologie di apprendimento

    Gli algoritmi di apprendimento e i paradigmi oggi utilizzati sono molteplici e in costante evoluzione/formulazione, ciascuno con velocità di convergenza sempre migliori (in base al compito svolto). Possiamo comunque ridurci a 3 grandi approcci (ciascuno dei quali ha eventualmente un suo sottotipo, ma non ci interessa per ora):
    - Apprendimento **supervisionato**: al computer vengono forniti dei dati di esempio e le etichette con cui devono essere correttamente identificati, con il fine ultimo di far sì che l'algoritmo deduca eventualmente come etichettare il set di dati in maniera autonoma;
    - Apprendimento **non supervisionato**: a differenza del primo, **non vi sono etichette fornite**. Di fatto, il compito dell'algoritmo è quello di *trovare autonomamente un pattern nel dataset*;
    - Apprendimento **per rinforzo**: l'algoritmo interagisce con un ambiente, venendo attivamente premiato o punito in base alle azioni che compie.

    **Nessuno dei 3 algoritmi** (così come i metodi matematici sfruttati in essi) **è *la* soluzione a tutti i problemi di apprendimento**, in quanto *dipende dal risultato desiderato dal modello di apprendimento e da ciò che si vuole fare con i dati*. Per fare un esempio per tutte le suddette categorie (ma di nuovo, senza scendere nel dettaglio dei diversi modelli matematici adottati),
    - l'apprendimento supervisionato è tendenzialmente usato generalmente in qualsiasi analisi dove è nota la caratteristica in esame e si vuole automatizzare questo processo di riconoscimento per dataset particolarmente grandi*. L'esempio semplice è quello di un insieme di immagini di animali pre-etichettati (quindi il set di dati *contiene già le risposte giuste*)
    - il non supervisionato serve a *studiare dataset nei quali si vogliano cercare caratteristiche comuni o raggruppare per caratteristica gruppi di dati* nei quali vi è qualche forma di correlazione statistica. Questo succede nei casi dove sono raccolte molte misure, ma non è possibile classificare in partenza cosa ogni dato possa rappresentare (perché si trattano ad esempio di dati puramente numerici senza un'etichetta sul singolo dato, come l'andamento della temperatura in un luogo, il prezzo di un bene in un intervallo di tempo o qualsiasi altro dato segua un andamento stocastico)
    - l'apprendimento per rinforzo è meglio utilizzato laddove l'algoritmo impara a risolvere *problemi decisionali più complessi*: la mossa migliore in un gioco, pilotare autonomamente un veicolo mantenendolo in corsia e reagendo agli ostacoli, qualsiasi problema il cui procedimento si riassume con "trovare la via corretta/più efficiente che soddisfi la richiesta".

    Per aggiungere ulteriore nomenclatura, abbiamo inoltre algoritmi *classificatori* e *regressori*:
    - un modello *classificatore* ha il compito di etichettare un certo dato come appartenente ad una *classe* (impara dunque a riconoscere a cosa si associa quel dato, l'esempio classico è quello di un set di misure dei petali e sepali di fiori, che un modello impara a distinguere nelle 3 specie);
    - un modello *regressore*, esattamente come il concetto di regressione statistica, fa una previsione sull'andamento dei dati associando una funzione continua ad un insieme discreto di misure: tali modelli servono dunque sia all'analisi statistica che alla previsione dell'andamento di dati successivi.

    Il confine tra i due non è per nulla netto: svariati modelli sono perfettamente capaci di svolgere entrambi i compiti, o se così non fosse, possono essere riadattati per svolgere l'altro compito.

    Infine, parliamo di modelli *parametrici* se l'apprendimento del modello è dettato, in buona parte o totalmente, da parametri che permettono di generalizzare a *qualsiasi altro dataset gli venga presentato*; sono altrimenti detti *non parametrici* se ogni istanza di apprendimento è autocontenuta, vale a dire che il modello è allenato da zero su ogni nuova esecuzione/nuovo dataset.

    # Le "features" e le "classi" di un set di dati

    In gergo, si chiamano *features* i valori di interesse associati ad un certo dato (in italiano, le sue *caratteristiche*): se si pensa all'esempio dei dataset di immagini, potremmo considerare misure di lunghezza e larghezza ed altri valori geometrici, colore, etc..., ma possono essere anche dei dati meno "concreti", ad esempio l'intensità di un segnale (luminoso, elettrico, sonoro...) in un intervallo di frequenza. Chiamiamo invece *classi* le etichette che vengono associate ai dati: ad esempio, abbiamo dei dati sulla lunghezza del pelo e delle orecchie di alcuni cani, e rileviamo che un certo dato appartiene alla *classe* dei *golden retriever*.

    Da un punto di vista più matematico e tecnico si ha una *matrice di dati*, le cui righe sono i singoli dati e le cui colonne sono le feature. Se $N$ sono i dati e $P$ le feature per ciascun dato, ci ritroviamo complessivamente con $N(P+1)$ valori (il +1 deriva dal considerare la colonna delle etichette in maniera separata dalle colonne di feature). Su questo dobbiamo fare sia delle considerazioni di carattere statistico che computazionale.

    ## Vincoli sulla dimensione dei dati

    In primis, perché un modello riesca ad apprendere in maniera ottimale, richiediamo che $N \gg P$: il motivo è che un algoritmo costretto ad operare su tante diverse caratteristiche con pochi esempi non sarà mai capace di generalizzare. In maniera qualitativa si richiede solitamente che *il numero di esempi sia 10 (talvolta 5) volte maggiore del numero di caratteristiche*: questa proporzione mitiga il fenomeno dell'*overfitting*, cioè quando il modello, invece di imparare ***dai dati***, ha *imparato proprio i dati*, performando in maniera molto scarsa quando gli vengono passati ulteriori dati nuovi.
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
