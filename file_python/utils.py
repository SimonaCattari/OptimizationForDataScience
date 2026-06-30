import os
import matplotlib.pyplot as plt

from config import (
    INSTANCE_PATH,
    EPSILON,
    MAX_ITER,
    TIME_TOL
)

from funzioni import (
    leggi_file_dimacs,
    genera_Q,
    FW_standard,
    AFW,
    solve_gurobi
)

# LOAD INSTANCE: n, m, u, b, q sono condivisi da tutte le run
n, m, u, b, q, edges, from_, to_ = leggi_file_dimacs(INSTANCE_PATH)



def run_fw(alpha, rho, seed, f_star):

    Q = genera_Q(alpha, u, q, m, rho, seed=seed)

    return FW_standard(
        n, m, u, b, from_, to_,
        Q, q,
        f_star = f_star,
        epsilon=EPSILON,
        max_iter=MAX_ITER,
        time_tol=TIME_TOL,
        visualize_res=False,
    )


def run_afw(alpha, rho, seed, f_star):

    Q = genera_Q(alpha, u, q, m, rho, seed=seed)

    return AFW(
        n, m, u, b, from_, to_,
        Q, q,
        f_star = f_star,
        epsilon=EPSILON,
        max_iter=MAX_ITER,
        time_tol=TIME_TOL,
        visualize_res=False
    )

def run_gurobi(alpha, rho, seed):
    # Genera la stessa Q usata da FW/AFW per il confronto (stesso alpha, p, seed)
    Q = genera_Q(alpha, u, q, m, rho, seed=seed)

    return solve_gurobi(
        n, m, u, b, from_, to_,
        Q, q
    )



def save_plot(name):

    folder = "results/plots"

    os.makedirs(folder, exist_ok=True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(folder, name),
        dpi=300
    )

    plt.close()