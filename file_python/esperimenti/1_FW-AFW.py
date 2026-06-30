"""
Esperimento 1 sezione 5.1: FW vs AFW
Confronto diretto tra Frank-Wolfe standard e Away-Steps Frank-Wolfe.
alpha=10, rho=0.0, seeds=[0,1,2] — singolo run su seed=0 per i plot.
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    run_fw,
    run_afw,
    run_gurobi,
    save_plot,
    genera_Q,
    n, m, u, b, q
)
from config import MAX_ITER

alpha = 10
rho = 0.0
SEEDS = [0, 1, 2]

# MULTI-SEED: statistiche tempi e iterazioni, per ciascun seed esegue Gurobi per ottenere f*, poi FW e AFW
fw_times = []
afw_times = []
fw_it = []
afw_it = []

for seed in SEEDS:
    # f* di riferimento per il calcolo del true relative gap
    x_star, f_star = run_gurobi(alpha, rho, seed)
    print(f"[seed={seed}] Gurobi f* = {f_star:.4e}")

    _, _, t_fw, g_fw, rel_fw, true_fw, k_fw, _, _, _ = run_fw(alpha, rho, seed, f_star=f_star)
    fw_times.append(t_fw[-1])
    fw_it.append(k_fw)

    _, _, t_afw, g_afw, rel_afw, true_afw, k_afw, _, active_set, n_drop, _ = run_afw(alpha, rho, seed, f_star=f_star)
    afw_times.append(t_afw[-1])
    afw_it.append(k_afw)


print("MULTIPLE SEEDS STATISTICS")
print(f"\nFW | mean time = {np.mean(fw_times):.4f}s ({np.std(fw_times):.2f}) | " f"mean iter = {np.mean(fw_it):.0f}") # 2.3391s (0.27) | mean iter = 1000
print(f"AFW | mean time = {np.mean(afw_times):.4f}s ({np.std(afw_times):.2f}) | " f"mean iter = {np.mean(afw_it):.0f}") # 7.3696s (2.27) | mean iter = 1000

# SINGLE RUN su seed=0 per i plot
seed = 0
x_star, f_star = run_gurobi(alpha, rho, seed)
print(f"\n[seed={seed}] Gurobi f* = {f_star:.4e}") # [seed = 0], 3.8305e+04

x_fw, f_fw, t_fw, g_fw, rel_fw, true_fw, k_fw, _, _, _ = run_fw(alpha, rho, seed, f_star=f_star)

x_afw, f_afw, t_afw, g_afw, rel_afw, true_afw, k_afw, _, active_set, n_drop, _ = run_afw(alpha, rho, seed, f_star=f_star)


print("FINAL SUMMARY (seed=0)")
print(f"\nGurobi f* = {f_star:.4e}") # 3.83 x 10^4
print(f"FW final rel gap = {rel_fw[-1]:.2e}") # 1.69 x 10^-4
print(f"AFW final rel gap = {rel_afw[-1]:.2e}") # 4.02 x 10^-5
if true_fw:
    print(f"FW true rel gap = {true_fw[-1]:.2e}") # 9.93 x 10^-5
if true_afw:
    print(f"AFW true rel gap = {true_afw[-1]:.2e}") # 5.22 x 10^-7
print(f"FW iterations = {k_fw}") # 1000
print(f"AFW iterations = {k_afw}") # 1000
print(f"FW time = {t_fw[-1]:.4f}s") # 1.87s
print(f"AFW time = {t_afw[-1]:.4f}s") # 9.10s
print(f"AFW drop steps = {n_drop}") # 12
print(f"AFW max active set = {max(active_set)}") # 158

# ANALISI TASSO DI CONVERGENZA
Q_seed0 = genera_Q(alpha, u, q, m, rho, seed=0)

# AFW: convergenza lineare
log_gap = np.log(true_afw[200:])
iters = np.arange(200, 200 + len(log_gap))
coeffs = np.polyfit(iters, log_gap, 1)
slope = coeffs[0]
rho_emp = 1 - np.exp(slope)

mu = 2 * Q_seed0[Q_seed0 > 0].min()
L = 2 * Q_seed0.max()
rho_th = mu / (mu + L)


print("CONVERGENCE RATE ANALYSIS")
print(f"slope empirica (AFW) = {slope:.6f}") # -0.003197
print(f"rho empirico  (AFW) = {rho_emp:.6f}") # 0.003192
print(f"mu (2*min Q) = {mu:.4e}") # 2.0801e-03
print(f"L (2*max Q) = {L:.4e}") # 1.1468e+01
print(f"rho teorico (AFW) = {rho_th:.6f}") # 0.000181
print(f"rapporto emp/teorico = {rho_emp/rho_th:.3f}x") # 17.603x
# python '1_FW-AFW.py' 2>&1 | tee output1 : output1.txt

# FW: convergenza sublineare O(1/t)
log_gap_fw = np.log(true_fw[10:])
log_iters = np.log(np.arange(10, 10 + len(log_gap_fw)))
coeffs_fw = np.polyfit(log_iters, log_gap_fw, 1)
slope_fw = coeffs_fw[0]
print(f"\nslope FW (log-log) = {slope_fw:.4f}  (teorico = -1.0)") # -0.9140

# Gap vs Iterazione
plt.figure(figsize=(8, 5))
plt.plot(true_fw, label="FW")
plt.plot(true_afw, label="AFW")

fit_iters = np.arange(200, len(true_afw))
fit_line = np.exp(coeffs[1] + coeffs[0] * fit_iters)
plt.plot(fit_iters, fit_line, '--', color='orange', label=f"AFW fit (p={rho_emp:.4f})")

h200 = true_afw[200]
theory_iters = np.arange(0, len(true_afw) - 200)
theory_line = h200 * (1 - rho_th) ** theory_iters
plt.plot(np.arange(200, len(true_afw)), theory_line, ':', color='red', label=f"Teorico worst-case (p={rho_th:.4f})")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("True Relative Gap")
plt.title(f"FW vs AFW | True Relative Gap vs Iteration | alpha={alpha} rho={rho}")
plt.legend()
plt.grid(True)
save_plot("fw_vs_afw_gap.png")

# Gap vs Tempo
plt.figure(figsize=(8, 5))
plt.plot(t_fw[:len(true_fw)],   true_fw,  label="FW")
plt.plot(t_afw[:len(true_afw)], true_afw, label="AFW")
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("True Relative Gap")
plt.title(f"FW vs AFW | Relative Gap vs Time | alpha={alpha} rho={rho}")
plt.legend()
plt.grid(True)
save_plot("fw_vs_afw_time.png")

# Valore Obiettivo vs Iterazione
plt.figure(figsize=(8, 5))
plt.plot(f_fw, label="FW")
plt.plot(f_afw, label="AFW")
plt.axhline(y=f_star, linestyle="--", label=f"Gurobi f* = {f_star:.4e}")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title(f"FW vs AFW | Objective | alpha={alpha} rho={rho}")
plt.legend()
plt.grid(True)
save_plot("fw_vs_afw_objective.png")

# Evoluzione Active Set AFW
plt.figure(figsize=(8, 5))
plt.plot(active_set)
plt.xlabel("Iteration")
plt.ylabel("Active Set Size")
plt.title(f"AFW Active Set Evolution | alpha={alpha} rho={rho}")
plt.grid(True)
save_plot("afw_active_set.png")
