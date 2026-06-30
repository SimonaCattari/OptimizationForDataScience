"""
Esperimento 4 sezione 6.1: Analisi con max_iter variabile:
    Scalabilità di FW e AFW su istanze più grandi (m=1000, 2000, 3000)
    alpha=10, rho=0.0, seed=0
    max_iter_FW=2000, max_iter_AFW=500 

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from funzioni import (
    leggi_file_dimacs,
    genera_Q,
    FW_standard,
    AFW,
    solve_gurobi
)
from config import EPSILON


ALPHA = 10
RHO = 0.0
SEED = 0

MAX_ITER_FW = 2000
MAX_ITER_AFW = 500
TIME_TOL_SCALABILITY = 300   


INSTANCES = {
    "m=1000": "1000/netgen-1000-1-2-a-b-s.dmx",
    "m=2000": "2000/netgen-2000-1-2-a-b-s.dmx",
    "m=3000": "3000/netgen-3000-1-2-a-b-s.dmx",
}

def first_true_gap_hit(true_gap, elapsed_time, eps):
    """
    Restituisce:
        iter_to_eps: int o None, indice prima iterazione con gap <= eps
        time_to_eps: float o None, tempo corrispondente a iter_to_eps
    """

    if not true_gap:
        return None, None

    for k, g in enumerate(true_gap):
        if g <= eps:
            t = elapsed_time[min(k, len(elapsed_time)-1)]
            return k, t
    return None, None 


def derive_status(iter_to_eps, raw_status):
    """
    Restituisce uno status leggibile che riflette la convergenza reale:
        - 'converged' se il true gap ha raggiunto epsilon 
        - il raw_status originale altrimenti
    """
    if iter_to_eps is not None:
        return "converged"
    return raw_status



rows = []
fw_traces = {}   
afw_traces = {}   

for label, path in INSTANCES.items():

    if not os.path.exists(path):
        print(f"[SKIP] {label}: file non trovato ({path})")
        continue

    print(f"\n{'='*60}")
    print(f"Istanza: {label}  ({path})")
    print(f"{'='*60}")

    # carica l'istanza e genera Q con i parametri fissati
    n, m, u, b, q, edges, from_, to_ = leggi_file_dimacs(path)
    Q = genera_Q(ALPHA, u, q, m, RHO, seed=SEED)
    print(f"nodi={n}  archi={m}")

    # f*
    print("[Gurobi] running...")
    x_star, f_star = solve_gurobi(n, m, u, b, from_, to_, Q, q)
    print(f"[Gurobi] f* = {f_star:.4e}")

    
    # FW
    print(f"[FW] running... (max_iter={MAX_ITER_FW})")
    (x_fw, f_fw, t_fw, g_fw,
     rel_fw, true_fw, k_fw,
     tempi_fw, _, status_fw) = FW_standard(
        n, m, u, b, from_, to_, Q, q,
        epsilon=EPSILON,
        max_iter=MAX_ITER_FW,
        time_tol=TIME_TOL_SCALABILITY,
        f_star=f_star,
        visualize_res=False
    )
    
    # Prima iterazione in cui il true gap scende sotto EPSILO
    fw_iter_to_eps, fw_time_to_eps = first_true_gap_hit(
        true_fw,
        t_fw,
        EPSILON
    )
    FW_status = derive_status(fw_iter_to_eps, status_fw)

    fw_time_to_eps = (
        t_fw[fw_iter_to_eps]
        if fw_iter_to_eps is not None
        else None
    )
    print(
        f"[FW] iter={k_fw} "
        f"iter_to_eps={fw_iter_to_eps} "
        f"time={t_fw[-1]:.2f}s "
        f"true_gap={true_fw[-1]:.2e} "
        f"status={FW_status}"
    )

    
    # AFW
    print(f"[AFW] running... (max_iter={MAX_ITER_AFW})")
    (x_afw, f_afw, t_afw, g_afw,
     rel_afw, true_afw, k_afw,
     tempi_afw, active_set, n_drop, status_afw) = AFW(
        n, m, u, b, from_, to_, Q, q,
        epsilon=EPSILON,
        max_iter=MAX_ITER_AFW,
        time_tol=TIME_TOL_SCALABILITY,
        f_star=f_star,
        visualize_res=False
    )
    afw_iter_to_eps, afw_time_to_eps = first_true_gap_hit(
        true_afw,
        t_afw,
        EPSILON
    )
    AFW_status = derive_status(afw_iter_to_eps, status_afw)

    afw_time_to_eps = (
        t_afw[afw_iter_to_eps]
        if afw_iter_to_eps is not None
        else None
    )
    print(
        f"[AFW] iter={k_afw} "
        f"iter_to_eps={afw_iter_to_eps} "
        f"time={t_afw[-1]:.2f}s "
        f"true_gap={true_afw[-1]:.2e} "
        f"drop_steps={n_drop} "
        f"status={AFW_status}"
    )

    fw_traces[label] = (t_fw,  true_fw)
    afw_traces[label] = (t_afw, true_afw)

    rows.append({
        "instance": label,
        "n_nodes": n,
        "n_arcs": m,
        "FW_iter": k_fw,
        "FW_iter_to_eps": fw_iter_to_eps,
        "FW_time_to_eps": fw_time_to_eps,
        "AFW_iter": k_afw,
        "AFW_iter_to_eps": afw_iter_to_eps,
        "AFW_time_to_eps": afw_time_to_eps,
        "FW_time": round(t_fw[-1],  3),
        "AFW_time": round(t_afw[-1], 3),
        "FW_true_gap": true_fw[-1]  if true_fw  else None,
        "AFW_true_gap": true_afw[-1] if true_afw else None,
        "FW_status": FW_status,
        "AFW_status": AFW_status,
        "AFW_n_drop": n_drop,
        "AFW_max_actset": max(active_set),
    })


if not rows:
    print("\nNessuna istanza trovata. Aggiorna i percorsi in INSTANCES.")
else:
    df = pd.DataFrame(rows)
    print("\n===== SUMMARY TABLE =====\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2e}" if isinstance(x, float) else str(x)))

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/scalability1.csv", index=False)
    print("\nSalvato in results/scalability1.csv")

    os.makedirs("results/plots", exist_ok=True)

    def save_plot(name):
        plt.tight_layout()
        plt.savefig(os.path.join("results/plots", name), dpi=300)
        plt.close()

    labels = df["instance"].tolist()
    x_ticks = range(len(labels))

    
    # tempo totale vs istanza, figura 6.3, crescita lineare FW e più che lineare di AFW al crescere di m
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, df["FW_time"],  marker="o", linewidth=2, label="FW")
    plt.plot(x_ticks, df["AFW_time"], marker="s", linewidth=2, label="AFW")
    plt.xticks(x_ticks, labels)
    plt.xlabel("Istanza (numero di archi)")
    plt.ylabel("Tempo totale (s)")
    plt.title(f"Scalabilità: tempo totale | alpha={ALPHA} rho={RHO}")
    plt.legend()
    plt.grid(True)
    save_plot("scalability_time1.png")

    # True Relative Gap vs Iterazione, figura 6.1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for lbl, (t, gap) in fw_traces.items():
        axes[0].plot(gap, linewidth=1.5, label=lbl)
    axes[0].axhline(y=EPSILON, color="black", linestyle="--", linewidth=1.2,
                    label=f"ε = {EPSILON:.0e}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iterazione")
    axes[0].set_ylabel("True Relative Gap")
    axes[0].set_title("FW — Gap vs Iterazione")
    axes[0].legend()
    axes[0].grid(True)

    for lbl, (t, gap) in afw_traces.items():
        axes[1].plot(gap, linewidth=1.5, label=lbl)
    axes[1].axhline(y=EPSILON, color="black", linestyle="--", linewidth=1.2,
                    label=f"ε = {EPSILON:.0e}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iterazione")
    axes[1].set_title("AFW — Gap vs Iterazione")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(
        f"True Relative Gap vs Iterazione | alpha={ALPHA} rho={RHO} "
        f"(FW budget={MAX_ITER_FW}, AFW budget={MAX_ITER_AFW})",
        fontsize=12
    )
    save_plot("scalability_gap_vs_iter1.png")

   
    # True Relative Gap vs Tempo, figura 6.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for lbl, (t, gap) in fw_traces.items():
        axes[0].plot(t[:len(gap)], gap, linewidth=1.5, label=lbl)
    axes[0].axhline(y=EPSILON, color="black", linestyle="--", linewidth=1.2,
                    label=f"ε = {EPSILON:.0e}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Tempo (s)")
    axes[0].set_ylabel("True Relative Gap")
    axes[0].set_title("FW — Gap vs Tempo")
    axes[0].legend()
    axes[0].grid(True)

    for lbl, (t, gap) in afw_traces.items():
        axes[1].plot(t[:len(gap)], gap, linewidth=1.5, label=lbl)
    axes[1].axhline(y=EPSILON, color="black", linestyle="--", linewidth=1.2,
                    label=f"ε = {EPSILON:.0e}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Tempo (s)")
    axes[1].set_title("AFW — Gap vs Tempo")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(
        f"True Relative Gap vs Tempo | alpha={ALPHA} rho={RHO}",
        fontsize=13
    )
    save_plot("scalability_gap_vs_time1.png")

    print("Plot salvati in results/plots/")
# python '4_scalability.py' 2>&1 | tee output4_variabile : output4_variabile.txt