"""
Algoritmi implementati:
    (A1a) Frank-Wolfe standard (FW) 
    (A1b) Away-Steps Frank-Wolfe (AFW) 
    (A2)  Gurobi Optimizer (solver di riferimento per validazione)
"""

import gc
import os
import csv
import time
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix

# LMO
from pyMCFSimplex import (MCFSimplex, new_darray, new_uiarray,
                          darray_get, uiarray_get,
                          CreateDoubleArrayFromList, CreateUIntArrayFromList)
import gurobipy as gp
from gurobipy import GRB #A2



# ECCEZIONE CUSTOM
class ProblemUnfeasibleError(Exception):
    pass


# LETTURA FILE DIMACS: Sezione 4
def leggi_file_dimacs(nome_file):
    """
    Legge un file in formato DIMACS (.dmx) e restituisce i parametri del
    problema di flusso a costo minimo lineare di base.

    Parametri
    ---------
    nome_file : file .dmx generato da netgen.

    Ritorna
    -------
    numero_nodi   : int
    numero_archi  : int
    u             : np.ndarray  : capacità superiori degli archi
    b             : list        : bilanci ai nodi (supply/demand)
    q             : np.ndarray  : costi lineari degli archi
    edges         : list of tuple
    from_         : list        : nodi di partenza (1-indexed)
    to_           : list        : nodi di arrivo   (1-indexed)
    """
    numero_nodi = 0
    numero_archi = 0
    u, b, q = [], [], []
    from_, to_, edges = [], [], []

    with open(nome_file, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'p':
                numero_nodi  = int(parts[2])
                numero_archi = int(parts[3])
                b = [0] * numero_nodi
            elif tag == 'n':
                b[int(parts[1]) - 1] = int(parts[2])
            elif tag == 'a':
                from_node = int(parts[1])
                to_node   = int(parts[2])
                from_.append(from_node)
                to_.append(to_node)
                u.append(int(parts[4]))
                q.append(int(parts[5]))
                edges.append((from_node, to_node))

    return numero_nodi, numero_archi, np.array(u, dtype=float), b, np.array(q, dtype=float), edges, from_, to_


# GENERAZIONE MATRICE Q: Sezione 4
def genera_Q(alpha, u, q, numero_archi, rho, seed=None):

    rng = np.random.default_rng(seed)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(u > 0, q / u, 0.0) #normalizzo il costo lineare sulla capacità dell'arco, q_i / u_i

    # estremi dell'intervallo di campionamento uniforme Qii \approx U [ (1-alpha/2) * ratio_i, (1+alpha/2) * ratio_i ]
    lb = (1.0 - alpha / 2.0) * ratio
    ub = (1.0 + alpha / 2.0) * ratio

    # gestione degli intervalli quando raio < 0 oppure alpha > 2
    lb_safe = np.minimum(lb, ub)
    ub_safe = np.maximum(lb, ub)

    Q_diag = rng.uniform(lb_safe, ub_safe)

    # troncamento a 0, garantisce Q semidefinita positiva Qii > 0, importante quando alpha > 2 perchè genera valori negativi
    Q_diag = np.maximum(Q_diag, 0.0)

    # p > 0: forza una frazioneegli elementi diagonali a zero, introduce archi a costo puramente lineare (perdita di stretta convessità)
    if rho > 0:
        n_zeri = int(np.round(rho * numero_archi))
        idx_zeri = rng.choice(numero_archi,
                               size=n_zeri,
                               replace=False)
        Q_diag[idx_zeri] = 0.0

    return Q_diag  # vettore diagonale


# LMO — Linear Minimization Oracle  
def _estrai_soluzione_mcf(mcf, numero_archi):
    """
    Estrae il vettore di flusso ottimo da un oggetto MCFSimplex già risolto.
    Restituisce np.ndarray di shape (numero_archi,).
    """
    length = mcf.MCFn()
    flow   = new_darray(mcf.MCFm())
    nms    = new_uiarray(length)
    mcf.MCFGetX(flow, nms)

    sol = np.zeros(numero_archi)
    for i in range(length):
        idx = uiarray_get(nms, i)
        if idx == 4294967295:   
            break
        sol[idx] = darray_get(flow, i)

    del flow, nms
    return sol

# Sezione 2.1, passo 3
def solve_LMO(n, numero_archi, u, obj_costs, b, from_, to_):
    """
    Risolve il sottoproblema lineare:
        s = argmin { obj_costs^T s : Es = b, 0 <= s <= u }
    tramite pyMCF-Simplex (Network Simplex).

    Usato sia per l'inizializzazione x0 (con obj_costs = q)
    che ad ogni iterazione FW/AFW (con obj_costs = gradiente).

    Lancia ProblemUnfeasibleError se il problema non è ottimo.
    """
    mcf = MCFSimplex()
    mcf.LoadNet(
        n, numero_archi, n, numero_archi,
        CreateDoubleArrayFromList(u.tolist()),
        CreateDoubleArrayFromList(obj_costs.tolist()),
        CreateDoubleArrayFromList([float(bi) for bi in b]),
        CreateUIntArrayFromList(to_),
        CreateUIntArrayFromList(from_)
    )
    mcf.SolveMCF()
    status_mcf = mcf.MCFGetStatus()

    sol = _estrai_soluzione_mcf(mcf, numero_archi) if status_mcf == 0 else None

    try:
        mcf.MCFFree()
    except AttributeError:
        pass

    if status_mcf != 0:
        raise ProblemUnfeasibleError(f"MCF infeasible (status={status_mcf})")

    return sol



# EXACT LINE SEARCH ANALITICA: Sexione 2.1, passo 4 e Sezione 2.2.2
def exact_line_search(x, d, gradient, Q, gamma_max):
    """
    Minimizza f(x + gamma * d) per gamma in [0, gamma_max] analiticamente.

    Per f(x) = x^T Q x + q^T x  con Q diagonale (passato come vettore 1D):
        gamma* = - grad^T d / (2 * d^T Q d)

    Se d^T Q d = 0 (funzione lineare lungo d), si assegna gamma = gamma_max.

    Ritorna (x_new, gamma_k).
    """
    dQd = float(d @ (Q * d)) # Q vettore diagonale
    if dQd <= 1e-15:
        gamma = gamma_max  # caso lineare lungo d: f è lineare, si prende il passo massimo
    else:
        # troncamento in [0, gamma_max] per mantenere l'ammissibilità
        gamma_star = -float(gradient @ d) / (2.0 * dQd)
        gamma = float(np.clip(gamma_star, 0.0, gamma_max))

    return x + gamma * d, gamma


# FRANK-WOLFE STANDARD: Sezione 3.1
def FW_standard(n, numero_archi, u, b, from_, to_,
                Q, q, epsilon, max_iter, f_star = None, time_tol=np.inf,
                visualize_res=False):
    """
    Frank-Wolfe standard:

    Passi (Algoritmo 1 del report):
      1. x0: LMO(q)             — vertice del politopo (BFS)
      2. grad: 2*Q*xk + q
      3. sk: LMO(grad)
      4. gk: grad^T (xk - sk) — FW Gap; stop se gk <= epsilon
      5. dFW: sk - xk
         gamma* = - grad^T dFW / (2 * dFW^T Q dFW)  [exact line search]
         gamma_k = clip(gamma*, 0, 1)
      6. xk+1: xk + gamma_k * dFW

    Parametri
    ---------
    n, numero_archi : dimensioni del grafo
    u, b, from_, to_: dati di rete
    Q: np.ndarray (m, m) diagonale PSD
    q: np.ndarray (m,) costi lineari
    epsilon: float, tolleranza sul FW gap
    max_iter: int, limite iterazioni
    time_tol: float, limite tempo (secondi)
    visualize_res: bool, stampa log per iterazione

    Ritorna
    -------
    x: np.ndarray, soluzione finale
    f_values: list, f(xk) ad ogni iterazione
    elapsed_time: list, tempo cumulato (secondi)
    dual_gap: list, FW gap ad ogni iterazione
    k: int, iterazioni effettuate
    tempi_per_it: list, tempo per singola iterazione
    active_set_size: list
    status: str
    """
    def f(x):
        return float(Q @ (x * x)) + float(q @ x)   # sum(Q_i * x_i^2) + q^T x

    # Passo 1: inizializzazione x0 tramite LMO
    start_global = time.perf_counter()
    x = solve_LMO(n, numero_archi, u, q, b, from_, to_)

    k = 0
    status = 'processing'
    dual_gap = []
    relative_gap = []
    true_relative_gap = []
    f_values = [f(x)]
    tempi_per_it = [time.perf_counter() - start_global]
    elapsed_time = [tempi_per_it[0]]

    while k < max_iter and elapsed_time[-1] < time_tol:
        t_start = time.perf_counter()

        # Passo 2: gradiente 
        gradient = 2.0 * (Q * x) + q

        # Passo 3: LMO sul gradiente corrente
        s_k = solve_LMO(n, numero_archi, u, gradient, b, from_, to_)

        # Passo 4: FW gap e criterio di arresto
        g_k = float(gradient @ (x - s_k))
        f_current = f(x)

        # relative gap normalizzato su f(x) per confrontare più istanze diverse
        rel_gap = g_k / max(1.0, abs(f_current))

        # true relative gap rispetto a Gurobi f*
        if f_star is not None:
            true_rel_gap = abs(f_current - f_star) / max(1.0, abs(f_star))
        else:
            true_rel_gap = None


        dual_gap.append(g_k)
        relative_gap.append(rel_gap)

        if true_rel_gap is not None:
            true_relative_gap.append(true_rel_gap)

        # criterio di arresto sul relative gap
        if rel_gap <= epsilon:
            status = 'found optimal'
            break

        # Passo 5: direzione FW eexact line search
        d_FW = s_k - x
        x, gamma_k = exact_line_search(x, d_FW, gradient, Q, gamma_max=1.0)

        if visualize_res:
            print(
                f"[FW] "
                f"iter={k:4d} | "
                f"f={f(x):.4e} | "
                f"relative gap={rel_gap:.2e} | "
                f"gamma={gamma_k:.2e} | "
                f"time={time.perf_counter()-t_start:.3f}s"
            )

        f_values.append(f(x))
        dt = time.perf_counter() - t_start
        tempi_per_it.append(dt)
        elapsed_time.append(elapsed_time[-1] + dt)

    if k == max_iter and status == 'processing':
        status = 'stopped (max_iter)'
    if elapsed_time[-1] >= time_tol and status == 'processing':
        status = 'stopped (time_tol)'

    if visualize_res:
        print(f'  FW  status: {status} | iter={k} | '
              f'f={f_values[-1]:.8e} | gap={dual_gap[-1] if dual_gap else 0:.4e}')

    active_set_size = [1] * len(f_values)
    return x, f_values, elapsed_time, dual_gap, relative_gap, true_relative_gap, k, tempi_per_it, active_set_size, status


# AWAY-STEPS FRANK-WOLFE: Sezione 3.2
def AFW(n, numero_archi, u, b, from_, to_,
        Q, q, epsilon, max_iter, f_star = None, time_tol=np.inf,
        visualize_res=False):
    """
    Away-Steps Frank-Wolfe:

    Passi (Algoritmo 2 del report):
      1. x0, S0 = {x0: 1.0}
      2. grad: 2*Q*xk + q
      3. sk: LMO(grad)
         ak: argmax_{v in Sk} grad^T v
      4. gk = grad^T (xk - sk);  stop se gk <= epsilon
      5. Se gk >= grad^T (ak - xk): 
            FW step, dFW = sk - xk, gamma_max = 1
            Altrimenti: Away step, dA  = xk - ak, gamma_max = λ_ak / (1 - λ_ak)
      6. gamma* = -grad^T d / (2 * d^T Q d)  [exact line search]
         gamma_k = clip(gamma*, 0, gamma_max)
      7. xk+1: xk + gamma_k * d
         Aggiornamento pesi e active set (Drop Step se gamma_k = gamma_max per away)

    Ritorna:
        x, f_values, elapsed_time, dual_gap, k, tempi_per_it, active_set_size, status
    """
    def f(x):
        return float(Q @ (x * x)) + float(q @ x)   # sum(Q_i * x_i^2) + q^T x

    # Passo 1: inizializzazione
    start_global = time.perf_counter()
    x = solve_LMO(n, numero_archi, u, q, b, from_, to_)

    # Active set dizionario ( {id: [vertice_atomo, peso_lambda]} )
    _aid_counter = [0]
    def _new_id():
        _aid_counter[0] += 1
        return _aid_counter[0]

    first_id = _new_id()
    S = {first_id: [np.array(x), 1.0]} # S0 = {x0}, lambda_x0 = 1

    k = 0
    status = 'processing'
    n_drop_steps = 0
    dual_gap = []
    relative_gap = []
    true_relative_gap = []
    f_values = [f(x)]
    tempi_per_it = [time.perf_counter() - start_global]
    elapsed_time = [tempi_per_it[0]]
    active_set_size = [1]

    while k < max_iter and elapsed_time[-1] < time_tol:
        t_start = time.perf_counter()

        # Passo 2: gradiente (Q vettore diagonale)
        gradient = 2.0 * (Q * x) + q

        # Passo 3: LMO e vertice away
        s_k = solve_LMO(n, numero_archi, u, gradient, b, from_, to_)

        # Vertice away: argmax_{v in S} grad^T v
        a_k_id  = max(S.keys(), key=lambda aid: float(S[aid][0] @ gradient))
        a_k     = S[a_k_id][0]
        lam_ak  = S[a_k_id][1]

        # Passo 4: FW gap e criterio di arresto
        g_k = float(gradient @ (x - s_k))
        f_current = f(x)

        # Relative FW gap
        rel_gap = g_k / max(1.0, abs(f_current))

        # True relative gap rispetto a Gurobi
        if f_star is not None:
            true_rel_gap = abs(f_current - f_star) / max(1.0, abs(f_star))
        else:
            true_rel_gap = None

        dual_gap.append(g_k)
        relative_gap.append(rel_gap)

        if true_rel_gap is not None:
            true_relative_gap.append(true_rel_gap)

        if visualize_res:
            print(f'  AFW iter={k:4d}  f={f_values[-1]:.8e}  rel_gap={rel_gap:.4e}  |S|={len(S)}')
       
        if rel_gap <= epsilon:
            status = 'found optimal'
            break

        # Passo 5: scelta direzione e gamma_max
        fw_decrement = g_k # grad^T (xk - sk)
        away_decrement = float(gradient @ (a_k - x)) # grad^T (ak - xk)

        if fw_decrement >= away_decrement: #FW step: ci muoviamo verso sk
            d_k = s_k - x
            gamma_max = 1.0
            is_fw_step = True
        else: # Away Step: ci allontaniamo dal vertice peggiore ak
            d_k = x - a_k
            gamma_max = lam_ak / (1.0 - lam_ak) if lam_ak < 1.0 - 1e-14 else np.inf
            is_fw_step = False

        # Passo 6: exact line search
        x, gamma_k = exact_line_search(x, d_k, gradient, Q, gamma_max)
        if visualize_res:
            print(
                f"[AFW] "
                f"iter={k:4d} | "
                f"f={f(x):.4e} | "
                f"relative gap={rel_gap:.2e} | "
                f"gamma={gamma_k:.2e} | "
                f"|S|={len(S):3d} | "
                f"time={time.perf_counter()-t_start:.3f}s"
        )
        # Passo 7: aggiornamento pesi e active set 
        if is_fw_step:
            if np.isclose(gamma_k, 1.0, atol=1e-12):
                # Drop all: reset completo
                new_id = _new_id()
                S = {new_id: [np.array(s_k), 1.0]} # gamma = 1: reset completo, xk+1 = sk -> active set = {sk}
            else:  # riscalamento pesi: tutti i lambda *= (1 - gamma_k)
                for aid in S:
                    S[aid][1] *= (1.0 - gamma_k)
                s_existing = next(
                    (aid for aid, (v, w) in S.items() if np.allclose(v, s_k, atol=1e-6)),
                    None
                )
                # sk già nell'active set -> aggiorna peso, altrimenti aggiunge
                if s_existing is not None:
                    S[s_existing][1] += gamma_k
                else:
                    new_id = _new_id()
                    S[new_id] = [np.asarray(s_k), gamma_k]
        else:
            # Away step: tutti i lambda *= (1 + gamma_k), tranne ak
            for aid in S:
                if aid != a_k_id:
                    S[aid][1] *= (1.0 + gamma_k)
            new_weight_ak = (1.0 + gamma_k) * lam_ak - gamma_k
            if np.isclose(gamma_k, gamma_max, atol=1e-10):
                # Drop step: gamma = gamma_max -> lambda_ak = 0, rimuove ak
                n_drop_steps += 1
                del S[a_k_id]  
                
            else:
                S[a_k_id][1] = new_weight_ak

        # Rimuove atomi con peso trascurabile (soglia 1e-5)
        # Poi rinormalizza per mantenere sum(lambda) = 1
        S = {aid: entry for aid, entry in S.items() if entry[1] > 1e-5}
        tot_w = sum(entry[1] for entry in S.values())
        if tot_w > 0:
            for entry in S.values():
                entry[1] /= tot_w

        f_values.append(f(x))
        active_set_size.append(len(S))

        MAX_ACTIVE_SET = 500
        if len(S) > MAX_ACTIVE_SET:
            status = "stopped (active set explosion)"
            break
        
        dt = time.perf_counter() - t_start
        tempi_per_it.append(dt)
        elapsed_time.append(elapsed_time[-1] + dt)
        k += 1

    if k == max_iter and status == 'processing':
        status = 'stopped (max_iter)'
    if elapsed_time[-1] >= time_tol and status == 'processing':
        status = 'stopped (time_tol)'

    if visualize_res:
        print(f'  AFW status: {status} | iter={k} | '
              f'f={f_values[-1]:.8e} | relative gap={relative_gap[-1] if relative_gap else 0:.4e} | '
              f'|S|={active_set_size[-1]}')
    if visualize_res:
        print(f"Total drop steps = {n_drop_steps}")

    return x, f_values, elapsed_time, dual_gap, relative_gap, true_relative_gap, k, tempi_per_it, active_set_size, n_drop_steps, status


# GUROBI 
def solve_gurobi(n, numero_archi, u, b, from_, to_, Q, q):
    from scipy.sparse import lil_matrix

    # Costruisce la matrice di incidenza E (n x m)
    # E[i,j] = +1 se arco j esce dal nodo i, -1 se entra, 0 altrimenti
    E = lil_matrix((n, numero_archi))
    for j in range(numero_archi):
        E[from_[j] - 1, j] = 1.0 # nodo di partenza
        E[to_[j]   - 1, j] = -1.0 # nodo di arrivo
    E = E.toarray()  

    model = gp.Model("MCF_QP")
    model.setParam("OutputFlag", 0)

    
    x_var = [model.addVar(lb=0.0, ub=u[i], name=f"x_{i}")
             for i in range(numero_archi)]
    x_arr = np.array(x_var)

    # Funzione obiettivo: x^T diag(Q) x + q^T x
    # np.diag(Q) ricostruisce la matrice da Q vettore diagonale
    model.setObjective(
        x_arr @ np.diag(Q) @ x_arr + q @ x_arr,
        sense=GRB.MINIMIZE
    )

    # vincoli si conservazione del flusso: Ex = b
    for i in range(n):
        model.addConstr(
            gp.quicksum(E[i, j] * x_var[j] for j in range(numero_archi)) == b[i]
        )

    model.optimize()
    model.setParam("OutputFlag", 0)

    
    bar_iter = model.getAttr("BarIterCount")
    sim_iter = model.getAttr("IterCount")
    method_used = "Barrier" if bar_iter > 0 else "Simplex/other"
    print(f"[Gurobi] metodo={method_used}  bar_iter={bar_iter}  simplex_iter={sim_iter}")

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi non ottimo (status={model.Status})")

    x_opt = np.array([v.X for v in x_var])
    f_opt = model.ObjVal  # f*

    model.dispose()
    return x_opt, f_opt