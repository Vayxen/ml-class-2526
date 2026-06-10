import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reti neurali artificiali

    ## Premessa biologica

    Storicamente non è uno dei più recenti modelli di classificazione: tuttavia è di fondamentale importanza poiché mira a riprodurre la funzione tipica del cervello (quali percezione di stimoli visivi e uditivi, il riconoscimento di pattern e forme, la comprensione del linguaggio, la coordinazione senso-motoria...). Si trattano fondamentalmente di astrazione matematica che provano appunto ad imitare/modellare in che modo il cervello risponde in natura ai suddetti stimoli.
    ***
    L'idea nasce già negli anni 40, dove si iniziò a modellare il "neurone artificiale". La cosa trova difficoltà tecniche e computazionali e riprende popolarità negli anni 80 con nuovi modelli di reti neurali. Seguirà il *deep learning* negli anni 90, che estende le ANN attraverso più strati di reti neurali, che rendono esponenzialmente più complesso il training (ma capace di apprendere informazioni di fatto più complesse).

    Si parte con un parallelo con il cervello umano: il cervello umano ha $10^{11}$ neuroni, costituiti da un corpo cellulare "centrale" e da varie ramificazioni (i dendriti) attraverso i quali ogni neurone comunica con gli altri attraverso segnali elettrici. Ogni neurone ha un ulteriore filamento detto *assone*, che si ramifica connettendosi ad altre cellule/altri dendriti. La comunicazione, a livello fisico, è allora dovuta ad un'eventuale differenza di potenziale (definisci *sinapsi*, e la differenza tra dendriti ed assoni). Seppur i segnali elettrici nei neuroni biologici siano più lenti di segnali trasmessi attraverso le componenti elettroniche, **queste connessioni si modificano nel tempo in base agli stimoli esterni, costituendo proprio la base dell'idea di "apprendimento"**. Gode inoltre di *fault tolerance*, vale a dire che il malfunzionamento/la morte di un neurone o alcune sue connessioni è danneggiata, non perde eccessivamente in performance. Il processo avviene di continuo giornalmente e chiaramente la performance del cervello con il passare degli anni degrada, tuttavia è un processo lungo a sufficienza da permetterci di funzionare bene fino alla vecchiaia (a scanso di patologie neurologiche quali demenza senile e Alzheimer).

    ## Introduzione alle reti neurali artificiali (ANNs)

    Riprodurre dunque il funzionamento del cervello richiede *modellare una rete di elementi che funzionano in parallelo, capace di apprendere e generalizzare*. Gli elementi in questione, detti *neuroni artificiali* (o unità, nodi, processori) fungono a mo' di *porta logica*:
    - sono caratterizzati da una variabile di stato $x_i$ (l'output del neurone)
    - un insieme di *sinapsi* $w_{ij}$, che sono gli "input" del neurone (pesati, da qui la notazione $w$)
    - un valore di soglia $\theta_i$, che detta a quale valore il neurone deve attivarsi (è talvolta detto *bias*)
    - la *funzione di attivazione*, che dunque prende le 3 caratteristiche sopra e stabilisce l'attivazione del neurone i-esimo in funzione dell'input da parte del neurone j-esimo ($i \neq j$), tenuta in considerazione il valore di soglia del neurone i-esimo. L'idea è quella dunque di una porta logica che si attiva quando riceve un segnale sufficientemente forte, sputando fuori un segnale di intensità variabile (*dettata dalla legge che definisce la funzione di attivazione*). **Non è necessario che tutti i neuroni della rete abbiano la stessa funzione di attivazione**: ad esempio, nelle reti neurali a più strati, si usa la funzione *softmax* nell'ultimo strato della rete.

    Lo stato del generico neurone è allora definito come

    $$
    x_i = f\left(\sum_{j=1}^N w_{ij}x_j-\theta_i\right)
    $$

    Possiamo anche modellare $\theta$ come il valore $w_{i0}$ incorporandolo nella sommatoria.

    $$
    x_i = f\left(\sum_{j=0}^N w_{ij}x_j\right)
    $$


    La configurazione più semplice che possiamo costruire sono i ***percettroni*** (reti neurali con un singolo strato): ad esempio vogliamo distinguere oggetti in due classi, rappresentandoli come punti nel piano. La retta di separazione allora è modellabile attraverso il singolo neurone, che si attiva o non si attiva (e dunque fa classificazione binaria in questo modo) in base al segnale (il dato nel piano). I parametri che stabiliscono la posizione della retta sono i pesi dei due ingressi + il "peso zero" dato dalla funzione di soglia, passati alla funzione di attivazione del neurone, che infine restituisce un valore in base alla sua legge. Questo concetto è generalizzabile ad $n$ classi, che allora costituiscono $n+1$ input per il percettrone.

    ## Problemi lineari e nonlineari

    I problemi lineari sono quindi ben facilmente risolti dai percettroni: tuttavia dovessimo trattare con problemi non lineari (come un problema modellato da uno XOR gate) **il singolo neurone non è più sufficiente a separare i semi-iperpiani (si ricorda: generalizzazione ad n dimensioni dei "semipiani/semirette") attraverso solo una semplice retta** (trattandosi di 4 punti nel piano, cioè gli stati $00, 11, 10, 01$). Da qui parte l'idea dei *multilayer perceptron* (**MLP**). (c'è altro da mettere)

    # Architettura di una NN

    In genere, una rete neurale "più complicata" del singolo percettrone è organizzata in *strati*. Gli strati sono sequenzialmente connessi, **senza connessioni intra-strato**: identifichiamo generalmente lo *strato input* (il primo strato in assoluto), lo *strato output* (l'ultimo, è spesso caratterizzato dalla funzione di attivazione *softmax*) e **gli strati intermedi, che vengono detti "nascosti"**. Senza scendere nel filosofico, sono detti tali poiché non è completamente noto il motivo o il significato dei valori restituiti da ciascun neurone in fase di addestramento/riconoscimento dei dati.


    Se tutte le connessioni sono dallo strato $n$ allo strato $n+1$, la rete si dice *feedforward*.

    Talvolta possono crearsi connessioni da uno strato successivo ad uno strato precedente, con lo scopo principale del ricalcolo/raffinamento dei pesi: si parla qui di *backpropagation*, letteralmente si tratta di propagazione all'indietro degli output di uno strato, così da migliorare iterativamente un output. Si potrebbe, per analogia, vedere come la rete che lavora su un input e, sulla base dell'output preso come nuovo input, torna indietro e "ritenta", nel tentativo di migliorare il risultato finale. Con un esempio pratico, potremmo considerare un algoritmo che impara a giocare a scacchi e prova tante mosse, prendendo i risultati di quelle mosse come nuovo input per capire se fare una specifica mossa sarà un guadagno o un errore alla lunga.

    > Laddove il singolo neurone di ciascun strato sia connesso con **tutti** i neuroni dello strato successivo, la rete si dice densa o completa.

    **Come è addestrata una rete neurale?** Ritornando ad un breve excursus neurobiologico, si evidenzia che "l'efficacia sinaptica di due neuroni aumenta laddove essi siano simultaneamente attivi" (*regola di Hebb*). Possiamo applicare questa regola al metodo di apprendimento della rete neurale, dove i pesi vengono iterativamente incrementati ogni volta che una coppia di neuroni si attiva insieme.

    ## Addestramento delta-rule

    > Altresì nota come *regola di Widrow-Hoff*.

    Quanto espresso sopra prende appunto il nome di *addestramento delta-rule*: il "delta" è la differenza tra l'output della rete e l'output desiderato. Sia $\vec{x}$ il vettore di input, $\vec{t}$ l'output voluto, $\vec{y}$ l'output effettivo della rete.

    Dato

    $$
    \delta_i = t_i-y_i
    $$

    Ciascun singolo peso varia in base a questa legge:

    $$
    \Delta w_i = \eta \delta x_i
    $$

    che viene poi sommato al peso $w_i$. $\eta$ è un parametro (arbitrario) tra 0 ed 1 che "pesa la variazione del peso", modificando in proporzione *solo i pesi delle connessioni che hanno avuto un input non nullo*).

    Parametro d'interesse è l'*errore quadratico della rete sul dato $k$, etichettato come *$E_k$,

    $$
    E_k = \frac{1}{2} \sum_i (t_i - y_i)^2
    $$

    Sommando su tutti i $k$ otteniamo *l'errore quadratico globale su tutto il set di training*.

    Ad inizio addestramento, i pesi hanno valori presi in maniera totalmente casuale. Geometricamente, possiamo vederlo come una posizione iniziale su una superficie (che rappresenta l'errore sopracitato) da minimizzare. In tal caso, il vettore dei pesi iterativamente cerca la direzione di massima pendenza (si parla qui di *gradient descent*)

    # Deep learning

    ## CNN

    ## Metodi e applicazioni del deep learning

    ### Transfer learning

    ### Feature extraction

    # Recurrent Neural Networks (RNNs)

    Di norma le NN sono entità prive di memoria: se invece vogliamo che l'algoritmo *tenga conto di stati precedenti quando deve prevedere il dato successivo*, si fa uso delle **reti neurali ricorrenti**. Questo modello è comune ad esempio nei LLMs (quali GPT/Claude/etc...) ma più semplicemente trova uso in casi come la previsione su tastiera del prossimo termine, o la traduzione, o il NLP (riscrivi meglio), anche generazione dati in base al passato (contesto più scientifico, o anche per cose come scacchi, etc)

    Nel pratico, la RNN contiene, come le NN normali, uno o più strati nascosti di neuroni. Preso un certo strato nascosto come vettore $h_t$, questo contiene i valori delle funzioni di attivazione (l'output) di ciascun neurone:

    $$
    h_t =
    \begin{pmatrix}
    h_1 \\
    h_2 \\
    \vdots \\
    \end{pmatrix}
    $$

    Strutturalmente, una RNN è costituita da una catena di questi strati nascosti/vettori, ai quali viene fornito un input $x$ con peso $U$. Lo strato di neuroni avrà un certo valore, che viene **ricorsivamente reinserito come input** (e pesato con peso $V).

    > Metti nelle righe precedenti: si addestra ad esempio una semplice NN feedforward, si trovano i vari $U_ij$ (trattasi dei pesi dallo strato input allo strato hidden) e i vari $W_ij$ (pesi dall'hidden all'output)

    In particolare,

    $$
    \begin{align*}
    h_t & = f(V_t h_{t-1} + U_t x_t + b_t) \\
    o_t & = f(W_t h_t + b_0)
    \end{align*}
    $$

    > inserisci meglio: i pesi possono rimanere costanti ($U,V,W$ rimangono uguali in qualsiasi step) o potrebbero essere successivamente ricalcolati (in tal caso inserisco la dipendenza dallo step $t$ (non $t-1$, perché tra un input e l'altro ricalcolo i pesi a partire da quelli precedenti, ergo $t-1$, e considero quelli nuovi come "pesi allo step $t$"))

    L'idea è che $h_t$ viene iterativamente ricalcolato considerando il suo valore allo step $t-1$. $b$ è un *termine di bias* (il valore di soglia che abbiamo considerato come "peso 0").
    """)
    return


if __name__ == "__main__":
    app.run()
