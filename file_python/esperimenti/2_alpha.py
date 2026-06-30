"""
Esperimento 2 sezione 5.2: Variazione di alpha
Effetto del condizionamento di Q sulle prestazioni di FW e AFW.
rho=0.0, alpha in {1, 5, 10}, seeds=[0,1,2] — mediana su 3 seed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    run_fw,
    run_afw,
    run_gurobi,
    save_plot
)
from config import MAX_ITER

alphas = [1, 5, 10]
rho = 0.0
SEEDS = [0, 1, 2]

summary_rows = []
fig_gap, axes_gap = plt.subplots(1, 2, figsize=(14, 5), sharey=True)


for alpha in alphas:
    print(f"\n{'='*55}\nalpha={alpha}\n{'='*55}")

    fw_gaps_all= []
    afw_gaps_all = []
    afw_dropsteps_all = []
    fw_final_gaps = []
    afw_final_gaps = []

    for seed in SEEDS:
        # f* di riferimento per il true relative gap
        x_star, f_star = run_gurobi(alpha, rho, seed)
        print(f"[seed={seed}] Gurobi f* = {f_star:.4e}")

        # FW
        _, _, t_fw, g_fw, rel_fw, true_fw, k_fw, _, _, _ = run_fw(alpha, rho, seed, f_star=f_star)
        fw_gaps_all.append(true_fw[:MAX_ITER])
        fw_final_gaps.append(true_fw[-1])

        # AFW
        _, _, t_afw, g_afw, rel_afw, true_afw, k_afw, _, active_set, n_drop, _ = run_afw(alpha, rho, seed, f_star=f_star)
        afw_gaps_all.append(true_afw[:MAX_ITER])
        afw_dropsteps_all.append(n_drop)
        afw_final_gaps.append(true_afw[-1])

        print(f"alpha={alpha}, seed={seed} | "
              f"FW final={true_fw[-1]:.2e} (iters={len(true_fw)}) | "
              f"AFW final={true_afw[-1]:.2e} (iters={len(true_afw)}) | "
              f"drop_steps={n_drop}")

    # Padding e mediana: i run possono finire a iterazioni diverse (convergenza o max_iter), per la mediana dobbiamo allinearli
    def pad_to(arr, length):
        arr = np.array(arr)
        if len(arr) < length:
            arr = np.concatenate([arr, np.full(length - len(arr), arr[-1])])
        return arr

    max_fw = max(len(g) for g in fw_gaps_all)
    max_afw = max(len(g) for g in afw_gaps_all)

    # mediana sui 3 seed
    median_fw_gap = np.median([pad_to(g, max_fw)  for g in fw_gaps_all],  axis=0)
    median_afw_gap = np.median([pad_to(g, max_afw) for g in afw_gaps_all], axis=0)

    summary_rows.append({
        "alpha": alpha,
        "FW gap mediano": np.median(fw_final_gaps),
        "AFW gap mediano": np.median(afw_final_gaps),
        "AFW gap std": np.std(afw_final_gaps),
        "AFW drop steps medi": np.mean(afw_dropsteps_all),
    })

    # curve mediane figura 5.5
    axes_gap[0].plot(median_fw_gap, linewidth=2, label=f"alpha={alpha}")
    axes_gap[1].plot(median_afw_gap, linewidth=2, label=f"alpha={alpha}")

# Gap vs Iterazione
for ax, title in zip(axes_gap, [f"FW | rho={rho}", f"AFW | rho={rho}"]):
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("True Relative Gap")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

fig_gap.suptitle(f"Effect of alpha on convergence (rho={rho}, median over {len(SEEDS)} seeds)", fontsize=13)
plt.tight_layout()
save_plot("effect_alpha_gap.png")


df = pd.DataFrame(summary_rows)
print("\nSUMMARY TABLE\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.2e}"))

# Drop Steps vs Alpha
plt.figure(figsize=(6, 4))
plt.plot(df["alpha"], df["AFW drop steps medi"], marker='o', linewidth=2)
plt.xlabel("alpha")
plt.ylabel("Average Drop Steps")
plt.title(f"Effect of conditioning on AFW drop steps (rho={rho})")
plt.grid(True)
save_plot("effect_alpha_dropsteps.png")

# python '2_alpha.py' 2>&1 | tee output2 : output2.txt