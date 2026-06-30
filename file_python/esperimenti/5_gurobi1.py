import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from funzioni import (
    leggi_file_dimacs,
    genera_Q,
    AFW,             
    solve_gurobi,
)

ALPHA = 10
RHO = 0.0
SEED = 0
EPSILON = 1e-8      
MAX_ITER = 10000     
TIME_TOL = 600       

INSTANCES = {
    "m=1000": r"1000/netgen-1000-1-2-a-b-s.dmx",
    "m=2000": r"2000/netgen-2000-1-2-a-b-s.dmx",
    "m=3000": r"3000/netgen-3000-1-2-a-b-s.dmx",
}

os.makedirs("results/plots", exist_ok=True)

rows = []
traces_afw = {}   
gurobi_times = {}   

for label, path in INSTANCES.items():

    if not os.path.exists(path):
        print(f"[SKIP] {label}: file non trovato ({path})")
        continue

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    # Carica istanza e genera Q con i parametri fissati
    n, m, u, b, q, edges, from_, to_ = leggi_file_dimacs(path)
    Q = genera_Q(ALPHA, u, q, m, RHO, seed=SEED)
    print(f"nodi={n}  archi={m}")

    # Esecuzione Gurobi
    print("[Gurobi] running ...")
    t0 = time.perf_counter()
    x_star, f_star = solve_gurobi(n, m, u, b, from_, to_, Q, q)
    t_gurobi = time.perf_counter() - t0
    gurobi_times[label] = t_gurobi
    print(f"[Gurobi] f*={f_star:.4e}  tempo={t_gurobi:.3f}s")

    # Esecuzione AFW
    print(f"[AFW] running ...")
    (x_afw, f_afw, elapsed_afw, dual_afw,
     rel_afw, true_afw, k_afw,
     tpi_afw, actset_afw, ndrop_afw, status_afw) = AFW(
        n, m, u, b, from_, to_, Q, q,
        epsilon=EPSILON,
        max_iter=MAX_ITER,
        time_tol=TIME_TOL,
        f_star=f_star,
        visualize_res=False,
    )
    print(f"[AFW] iter={k_afw}  tempo={elapsed_afw[-1]:.2f}s  "
          f"true_gap={true_afw[-1]:.2e}  status={status_afw}")

    # Allinea elapsed_time e true_relative_gap
    def align(elapsed, true_gap):
        n_gap = len(true_gap)
        return elapsed[-n_gap:], true_gap

    traces_afw[label] = align(elapsed_afw, true_afw)

    rows.append({
        "Istanza": label,
        "Gurobi tempo (s)": round(t_gurobi, 3),
        "AFW tempo (s)": round(elapsed_afw[-1], 3),
        "AFW iter": k_afw,
        "AFW true gap": true_afw[-1] if true_afw else np.nan,
        "AFW status": status_afw,
        "AFW / Gurobi": round(elapsed_afw[-1] / t_gurobi, 1),
    })

if not rows:
    print("\nNessuna istanza trovata. Aggiorna i percorsi in INSTANCES.")
else:
    df = pd.DataFrame(rows)
    print("\n===== TABELLA CONFRONTO FAIR =====")
    print(df.to_string(index=False))
    df.to_csv("results/gurobi_gf.csv", index=False)
    print("\nSalvato in results/gurobi_gf.csv")

    
    # True relative gap vs Tempo
    n_inst = len(traces_afw)
    fig, axes = plt.subplots(1, n_inst, figsize=(6 * n_inst, 5), sharey=False)
    if n_inst == 1:
        axes = [axes]

    for ax, label in zip(axes, traces_afw.keys()):
        t_afw, h_afw = traces_afw[label]
        t_gur = gurobi_times[label]

        ax.semilogy(t_afw, h_afw, color="darkorange", linewidth=1.8, label="AFW")
        ax.axvline(t_gur, color="green", linestyle="--", linewidth=1.5,
                   label=f"Gurobi ({t_gur:.2f}s)")
        ax.axhline(EPSILON, color="grey", linestyle=":", linewidth=1.2,
                   label=f"ε = {EPSILON:.0e}")

        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Tempo (s)", fontsize=11)
        ax.set_ylabel("True relative gap", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.4)

    fig.suptitle(
        f"True relative gap vs Tempo — Confronto Fair (AFW vs Gurobi)\n"
        f"(α={ALPHA}, ρ={RHO}, seed={SEED}, ε={EPSILON})",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig("results/plots/gf_comparison_gap_vs_time.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Grafico salvato: results/plots/gf_comparison_gap_vs_time.png")

    
    # Bar chart per il target di 1e-8
    GAP_TARGET = EPSILON 

    def time_to_target(elapsed, gaps, target):
        """Primo tempo in cui true_gap scende sotto target. None se mai."""
        for t, g in zip(elapsed, gaps):
            if g <= target:
                return t
        return None

    labels_plot = df["Istanza"].tolist()
    x_pos = np.arange(len(labels_plot))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {"Gurobi": "green", "AFW": "darkorange"}

    for i, label in enumerate(labels_plot):
        t_gur = gurobi_times[label]
        t_afw_arr, h_afw_arr = traces_afw[label]

        entries = [
            ("Gurobi", t_gur, t_gur, 0.0), 
            ("AFW", None, time_to_target(t_afw_arr, h_afw_arr, GAP_TARGET), h_afw_arr[-1]),
        ]

        for j, (name, t_total, t_tgt, gap_final) in enumerate(entries):
            xpos = i + (j - 0.5) * width
            color = colors[name]

            if t_tgt is not None:
                bar = ax.bar(xpos, t_tgt, width, color=color, alpha=0.85,
                             label=name if i == 0 else "_nolegend_")
                ax.bar_label(bar, labels=[f"{t_tgt:.1f}s"],
                             padding=2, fontsize=8)
            else:
                # Non ha raggiunto il target 
                t_end = t_afw_arr[-1]
                bar = ax.bar(xpos, t_end, width, color=color, alpha=0.35,
                             hatch="//",
                             label=(name + " (target non raggiunto)")
                             if i == 0 else "_nolegend_")
                ax.annotate(
                    f"gap={gap_final:.0e}",
                    xy=(xpos, t_end),
                    xytext=(xpos, t_end + 0.5),
                    ha="center", va="bottom", fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_plot)
    ax.set_xlabel("Istanza", fontsize=11)
    ax.set_ylabel("Tempo (s)", fontsize=11)
    ax.set_title(
        f"Tempo per raggiungere true gap ≤ {GAP_TARGET:.0e} (Confronto Fair)\n"
        f"(α={ALPHA}, ρ={RHO}, seed={SEED})",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig("results/plots/gf_comparison_total_time.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Grafico salvato: results/plots/gf_comparison_total_time.png")
# python '5_gurobi1.py' 2>&1 | tee output_gf : output_gf.txt