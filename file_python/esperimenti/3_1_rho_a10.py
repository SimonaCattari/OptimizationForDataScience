import numpy as np
import matplotlib.pyplot as plt

from utils import (
    run_afw,
    run_fw,
    run_gurobi,
    save_plot
)

alpha = 10
SEEDS = [0, 1, 2]
rhos = [0.0, 0.25, 0.50, 0.75]
EPSILON = 1e-6
YLIM = (1e-11, 1e2)


def pad_to(arr, target_len):
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.full(target_len - len(arr), arr[-1])])



# Raccolta dati
MAX_LEN_FW = 0
MAX_LEN_AFW = 0

all_fw_raw = {rho: [] for rho in rhos}
all_afw_raw = {rho: [] for rho in rhos}

for rho in rhos:
    for seed in SEEDS:
        x_star, f_star = run_gurobi(alpha, rho, seed)

        _, _, t_fw, g_fw, rel_fw, true_fw, k_fw, _, _, _ = \
            run_fw(alpha, rho, seed, f_star=f_star)

        _, _, t_afw, g_afw, rel_afw, true_afw, k_afw, _, active_set, n_drop, _ = \
            run_afw(alpha, rho, seed, f_star=f_star)

        all_fw_raw[rho].append(np.array(true_fw))
        all_afw_raw[rho].append(np.array(true_afw))

        MAX_LEN_FW  = max(MAX_LEN_FW,  len(true_fw))
        MAX_LEN_AFW = max(MAX_LEN_AFW, len(true_afw))



# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for rho in rhos:
    # FW 
    min_fw = min(len(g) for g in all_fw_raw[rho])
    median_fw = np.median([g[:min_fw] for g in all_fw_raw[rho]], axis=0)
    axes[0].plot(median_fw, linewidth=2, label=f"rho={rho}")

    # AFW: pad al massimo del gruppo, calcola mediana, taglia al minimo
    max_afw_rho = max(len(g) for g in all_afw_raw[rho])
    padded = np.array([pad_to(g, max_afw_rho) for g in all_afw_raw[rho]])
    median_afw = np.median(padded, axis=0)

    idx_min = np.argmin(median_afw)
    axes[1].plot(median_afw[:idx_min + 1], linewidth=2, label=f"rho={rho}")

# Linea epsilon
for ax in axes:
    ax.axhline(y=EPSILON, color="gray", linestyle="--", linewidth=1.2,
               label="$\\varepsilon$ = 10$^{-6}$")

# Formattazione FW
axes[0].set_yscale("log")
axes[0].set_ylim(YLIM)
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("True Relative Gap")
axes[0].set_title(f"Effect of rho on FW | alpha={alpha}")
axes[0].legend(loc="upper right")
axes[0].grid(True)

# Formattazione AFW
axes[1].set_yscale("log")
axes[1].set_ylim(YLIM)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("True Relative Gap")
axes[1].set_title(f"Effect of rho on AFW | alpha={alpha}")
axes[1].legend(loc="upper right")
axes[1].grid(True)

fig.suptitle(
    f"Effect of rho on convergence (alpha={alpha}, median over {len(SEEDS)} seeds)",
    fontsize=13
)

plt.tight_layout()
save_plot("effect_rho_a10.png")
