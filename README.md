# Min-Cost Flow Problem — Frank-Wolfe e Away-Steps Frank-Wolfe

Progetto per il corso di **Optimization for Data Science** (Data Science and Business Informatics, Università di Pisa, A.A. 2025/2026).

## Panoramica

Il progetto risolve un problema di **flusso a costo minimo con funzione obiettivo quadratica convessa e separabile**:

```
min  xᵀQx + qᵀx
s.t. Ex = b,  0 ≤ x ≤ u
```

dove `E` è la matrice di incidenza nodi-archi di un grafo orientato, `b` il vettore dei bilanci ai nodi, `u` le capacità degli archi, `q` i costi lineari e `Q` una matrice diagonale semidefinita positiva dei coefficienti quadratici.

Il problema viene risolto tramite due metodi del **Gradiente Condizionale (Frank-Wolfe)**, che evitano le costose proiezioni ortogonali sul politopo ammissibile sostituendole, ad ogni iterazione, con la risoluzione di un sottoproblema di flusso a costo minimo *lineare* tramite l'algoritmo **Network Simplex** (`pyMCF-Simplex`):

- **Frank-Wolfe (FW)** — metodo del gradiente condizionale standard con exact line search
- **Away-Steps Frank-Wolfe (AFW)** — variante di Lacoste-Julien e Jaggi (2015) che mantiene un active set dei vertici visitati e alterna passi FW e passi "away" per evitare il fenomeno dello zig-zagging, recuperando una **convergenza lineare globale** sui politopi

Entrambe le implementazioni sono validate confrontandole con il solver commerciale **Gurobi** (tramite `gurobipy`) sulla formulazione quadratica originale.

## Metodologia

1. **Inizializzazione** — `x₀` si ottiene risolvendo il problema di flusso a costo minimo lineare sui costi originali `q`, garantendo una soluzione ammissibile di base (vertice del politopo) fin dal primo passo.
2. **Linearizzazione** — ad ogni iterazione viene calcolato il gradiente `∇f(xₖ) = 2Qxₖ + q` (operazione efficiente grazie alla struttura separabile).
3. **Oracolo di Ottimizzazione Lineare (LMO)** — viene risolto un problema di flusso a costo minimo lineare sul gradiente corrente tramite Network Simplex, per trovare il vertice FW `sₖ`.
4. **(solo AFW) Direzione Away** — viene identificato il vertice "peggiore" attualmente nell'active set per calcolare una direzione away `d_A`; l'algoritmo sceglie la direzione (FW o away) che garantisce la maggiore riduzione potenziale.
5. **Exact line search** — il passo ottimo viene calcolato analiticamente (forma chiusa per funzioni quadratiche) e troncato per preservare l'ammissibilità.
6. **Criterio di arresto** — viene usato il **Frank-Wolfe gap** (duality gap), un limite superiore rigoroso all'errore di ottimalità per funzioni convesse, come certificato di arresto.

## Setup sperimentale

- Istanze lineari della rete generate con **netgen** (formato DIMACS), dalla collezione *Single-commodity NonLinear Network Design*, con 1000, 2000 e 3000 archi.
- La componente quadratica `Q` viene generata separatamente, campionata attorno a `qᵢ/uᵢ`, controllata da due parametri:
  - `α` — controlla il condizionamento di `Q` (e quindi `L/μ`)
  - `ρ` — frazione di archi forzati ad avere costo puramente lineare (`Qᵢᵢ = 0`)
- Ogni configurazione è ripetuta su più seed casuali per garantire robustezza statistica.
- Solver di riferimento: **Gurobi**, eseguito fino al soddisfacimento dei criteri di ottimalità interni (metodo Barrier).

## Risultati principali

- **FW vs AFW**: AFW raggiunge true relative gap significativamente più contenuti (fino a diversi ordini di grandezza inferiori) rispetto a FW standard, al costo di un overhead per iterazione maggiore dovuto alla gestione dell'active set.
- **Tasso di convergenza**: empiricamente, la convergenza di AFW risulta sensibilmente più rapida del tasso teorico worst-case `ρ_th = μ/(μ+L)`, poiché quest'ultimo ignora la geometria favorevole del politopo.
- **Effetto del condizionamento (α)**: FW è relativamente insensibile ad `α` (la sua dinamica è dominata dallo zig-zagging), mentre la convergenza di AFW — e il numero di drop steps — è fortemente influenzata dal numero di condizionamento.
- **Effetto degli archi a costo lineare (ρ)**: valori elevati di `ρ` innescano una rapida *manifold identification* in AFW (convergenza quasi istantanea), `ρ = 0` beneficia della garanzia di convergenza lineare, mentre valori intermedi (es. 0.25) ricadono in una "zona di transizione" in cui nessuno dei due meccanismi opera pienamente.
- **Scalabilità**: AFW scala bene fino a 3000 archi, ma la crescita dell'active set può diventare un collo di bottiglia pratico (es. salvaguardia per *active set explosion* attivata sull'istanza più grande).
- **Confronto con Gurobi**: a precisione moderata (10⁻⁴), AFW è nettamente più veloce sia di FW sia di Gurobi (fino a ~60 volte più veloce di Gurobi sull'istanza più grande). Ad alta precisione (10⁻⁸), AFW resta competitivo ma è più sensibile alla dimensione dell'active set e alla lenta riduzione del FW gap nelle fasi finali.

## Struttura del repository

```
OPT_project39Cattari-Trivelli/
├── file_python/
│   ├── esperimenti/
│   │   ├── 1_FW-AFW.py         # Sezione 5.1 — confronto diretto FW vs AFW
│   │   ├── 2_alpha.py          # Sezione 5.2 — effetto del parametro alpha
│   │   ├── 3_rho_a1.py         # Sezione 5.3.1 — effetto del parametro rho (alpha=1)
│   │   ├── 3_1_rho_a10.py      # Sezione 5.3.2 — effetto del parametro rho (alpha=10)
│   │   ├── 4_scalability.py    # Sezione 6.1 — scalabilità su m = 1000/2000/3000
│   │   ├── 5_gurobi.py         # Sezione 6.3.1 — confronto con Gurobi (precisione moderata)
│   │   └── 5_gurobi1.py        # Sezione 6.3.2 — confronto con Gurobi (alta precisione)
│   ├── config.py                # Parametri globali: istanza da caricare, EPSILON, MAX_ITER, TIME_TOL
│   ├── utils.py                 # Caricamento istanza e wrapper run_fw, run_afw, run_gurobi, save_plot
│   └── funzioni.py              # Lettura DIMACS, generazione di Q, LMO (pyMCFSimplex), line search,
│                                 # Frank-Wolfe standard, Away-Steps Frank-Wolfe, solve_gurobi
├── 1000/ 2000/ 3000/            # Istanze NETGEN (.dmx) raggruppate per numero di archi
├── results/                     # Output degli esperimenti
│   ├── plots/                   # Grafici (.png) usati nel report
│   └── *.csv                    # Tabelle riassuntive
└── output_terminale/            # Log di esecuzione degli esperimenti
```


## Strumenti utilizzati

- Python
- `pyMCF-Simplex` (Network Simplex come LMO)
- `gurobipy` (Gurobi Optimizer, per la validazione)
- `netgen` (generazione delle istanze di test, formato DIMACS)
- Matplotlib (grafici di convergenza e scalabilità)

## Riferimenti bibliografici

- S. Lacoste-Julien, M. Jaggi. *On the Global Linear Convergence of Frank-Wolfe Optimization Variants*. NeurIPS, 2015.
- I. M. Bomze, F. Rinaldi, D. Zeffiro. *Active Set Complexity of the Away-step Frank-Wolfe Algorithm*. arXiv:1912.11492, 2019.
- M. Jaggi. *Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization*. ICML, 2013.
- J. B. Orlin. *A polynomial time primal network simplex algorithm for minimum cost flows*. Mathematical Programming, 1997.
- P. Kovács. *Minimum-cost flow algorithms: An experimental evaluation*. Optimization Methods and Software, 2015.

## Autori

- Cattari Simona
- Trivelli Matteo



