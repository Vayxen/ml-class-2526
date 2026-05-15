import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reti neurali artificiali

    Storicamente non è uno dei più recenti modelli di classificazione: tuttavia è di fondamentale importanza poiché mira a riprodurre la funzione tipica del cervello (quali percezione di stimoli visivi e uditivi, il riconoscimento di pattern e forme, la comprensione del linguaggio, la coordinazione senso-motoria...). Si trattano fondamentalmente di astrazione matematica che provano appunto ad imitare/modellare in che modo il cervello risponde in natura ai suddetti stimoli.
    ***
    L'idea nasce già negli anni 40, dove si iniziò a modellare il "neurone artificiale". La cosa trova difficoltà tecniche e computazionali e riprende popolarità negli anni 80 con nuovi modelli di reti neurali. Seguirà il *deep learning* negli anni 90, che estende le ANN attraverso più strati di reti neurali, che rendono esponenzialmente più complesso il training (ma capace di apprendere informazioni di fatto più complesse).

    Si parte con un parallelo con il cervello umano: il cervello umano ha $10^{11}$ neuroni, costituiti da un corpo cellulare "centrale" e da varie ramificazioni (i dendriti) attraverso i quali ogni neurone comunica con gli altri attraverso segnali elettrici. Ogni neurone ha un ulteriore filamento detto *assone*, che si ramifica connettendosi ad altre cellule/altri dendriti. La comunicazione, a livello fisico, è allora dovuta ad un'eventuale differenza di potenziale (definisci *sinapsi*, e la differenza tra dendriti ed assoni). Seppur i segnali elettrici nei neuroni biologici siano più lenti di segnali trasmessi attraverso le componenti elettroniche, **queste connessioni si modificano nel tempo in base agli stimoli esterni, costituendo proprio la base dell'idea di "apprendimento"**. Gode inoltre di *fault tolerance*, vale a dire che il malfunzionamento/la morte di un neurone o alcune sue connessioni è danneggiata, non perde eccessivamente in performance. Il processo avviene di continuo giornalmente e chiaramente la performance del cervello con il passare degli anni degrada, tuttavia è un processo lungo a sufficienza da permetterci di funzionare bene fino alla vecchiaia (a scanso di patologie neurologiche quali demenza senile e Alzheimer).

    Riprodurre dunque il funzionamento del cervello richiede *modellare una rete di elementi che funzionano in parallelo, capace di apprendere e generalizzare*. Gli elementi in questione, detti *neuroni artificiali* (o unità, nodo, processore) funge a mo' di *porta logica*:
    - è caratterizzato da una variabile di stato $x_i$ (l'output del neurone)
    - un insieme di *sinapsi* $w_{ij}$, che sono gli "input" del neurone (pesati, da qui la notazione $w$)
    - un valore di soglia $\theta_i$, che detta a quale valore il neurone deve attivarsi (è talvolta detto *bias*)
    - la *funzione di attivazione*, che dunque prende le 3 caratteristiche sopra e stabilisce l'attivazione del neurone i-esimo in funzione dell'input da parte del neurone j-esimo (i!=j), tenuta in considerazione il valore di soglia del neurone i-esimo. L'idea è quella dunque di una porta logica che si attiva quando riceve un segnale sufficientemente forte, sputando fuori un segnale di intensità variabile (dettata dalla legge della funzione di attivazione).

    Lo stato del generico neurone è allora definito come

    $$
    x_i = f(\sum_{i=1})
    $$

    Possiamo anche modellare $\theta$ come il valore $w_{i0}$ incorporandolo nella sommatoria:
    $$
    balls
    $$

    Le funzioni di attivazione comunemente scelte sono la *funzione a gradino* o le *sigmoidi*.

    La configurazione più semplice che possiamo costruire sono i ***percettroni*** (reti neurali con un singolo strato): ad esempio vogliamo distinguere oggetti in due classi, rappresentandoli come punti nel piano. La retta di separazione allora è modellabile attraverso il neurone, che si attiva o non si attiva (e dunque fa classificazione binaria in questo modo) in base al segnale (il dato nel piano). I parametri che stabiliscono allora la posizione della retta sono i pesi dei due ingressi + il "peso zero" dato dalla funzione di soglia, passati alla funzione di attivazione del neurone, che allora restituisce un valore in base alla sua legge. Questo concetto è generalizzabile ad $n$ classi, che allora costituiscono $n+1$ input per il percettrone.

    ## Problemi lineari e nonlineari

    I problemi lineari sono quindi ben facilmente risolti dai percettroni: tuttavia dovessimo trattare con problemi non lineari (come un problema modellato da uno XOR gate) **il singolo neurone non è più sufficiente a separare i semi-iperpiani attraverso solo una semplice retta** (trattandosi di 4 punti nel piano, cioè gli stati $00, 11, 10, 01$). Da qui parte l'idea dei *multilayer perceptron*.
    """)
    return


if __name__ == "__main__":
    app.run()
