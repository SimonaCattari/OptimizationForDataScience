Progetto 39 — Cattari, Trivelli
Implementazione di Frank-Wolfe (FW) e Away-Steps Frank-Wolfe (AFW) per il Min-Cost Flow quadratico separabile.

Struttura della cartella:
OPT_project39Cattari-Trivelli/

 file_python/        

	- esperimenti

		-> 1_FW-AFW.py: Esperimento sezione 5.1 — confronto diretto FW vs AFW 

		-> 2_alpha.py: Esperimento sezione 5.2 — effetto del parametro alpha

		-> 3_rho_a1.py/3_1_rho_a10.py: Esperimento sezione 5.3.1/5.3.2 sul parametro rho (con alpha=1 e alpha=10)

		-> 4_scalability.py: Esperimento sezione 6.1 — scalabilità sum = 1000/2000/3000 |

		-> 5_gurobi.py/5_gurobi1.py: Esperimento sezione 6.3.1/6.3.2 sul confronto con Gurobi 

	- config.py: parametri globali; istanza da caricare, EPSILON, MAX_ITER, TIME_TOL

	- utils.py: carica l'istanza una sola volta e fornisce i wrapper run_fw, run_afw, run_gurobi, save_plot

	- funzioni.py: lettura DIMACS, generazione di Q, LMO (pyMCFSimplex), line search, FW standard, Away-Steps Frank Wolfw, solve_gurobi

 1000/ 2000/ 3000/ (istanze NETGEN (.dmx) raggruppate per numero di archi)
 results/  (output degli esperimenti)

	- plots/  (grafici (.png) usati nel report)
	- .csv  (tabelle riassuntive)

 output_terminale/ 

