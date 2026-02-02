"""
Évaluation comparative des algorithmes d'inférence selon la taille du graphe et la force de couplage.

Enregistre les résultats CSV dans outputs/csv/ pour la génération de graphiques.

Auteur : Mouhssine Rifaki
"""

import time
import csv
import os
import numpy as np
from src.belief_propagation import sum_product_bp, loopy_bp
from src.variational import mean_field_vi
from src.trw import trw_s
from src.utils import generate_ising_grid, generate_chain
from src.stereo import run_stereo_experiment


def convergence_vs_graph_size(n_runs=3):
    """Mesure la vitesse de convergence de la BP bouclée, TRW-S et champ moyen
    sur des grilles d'Ising.

    Parcourt les tailles de grille de 3x3 à 10x10 et enregistre :
      - le nombre d'itérations pour converger
      - le temps d'exécution
    """
    sizes = [3, 4, 5, 6, 7, 8, 10]
    results = []

    for sz in sizes:
        bp_iters, bp_times = [], []
        trw_iters, trw_times = [], []
        mf_iters, mf_times = [], []

        for run in range(n_runs):
            fg, _ = generate_ising_grid(sz, sz, coupling=0.5, field=0.1, seed=run)

            # BP bouclée
            t0 = time.time()
            _, bp_conv, bp_hist = loopy_bp(fg, damping=0.5, max_iter=300, tol=1e-6)
            bp_times.append(time.time() - t0)
            bp_iters.append(len(bp_hist))

            # TRW-S
            t0 = time.time()
            _, trw_conv, trw_hist = trw_s(fg, damping=0.5, max_iter=300, tol=1e-6)
            trw_times.append(time.time() - t0)
            trw_iters.append(len(trw_hist))

            # Inférence variationnelle par champ moyen
            t0 = time.time()
            _, mf_elbo, mf_conv = mean_field_vi(fg, max_iter=300, tol=1e-6, seed=run)
            mf_times.append(time.time() - t0)
            mf_iters.append(len(mf_elbo))

        results.append({
            "grid_size": sz,
            "n_variables": sz * sz,
            "bp_iters": np.mean(bp_iters),
            "bp_time_s": np.mean(bp_times),
            "trw_iters": np.mean(trw_iters),
            "trw_time_s": np.mean(trw_times),
            "mf_iters": np.mean(mf_iters),
            "mf_time_s": np.mean(mf_times),
        })

    return results


def accuracy_vs_coupling(n_runs=3):
    """Compare les croyances BP, TRW-S et champ moyen aux marginales exactes
    sur de petites chaînes.

    Les marginales exactes sont calculées par la distribution jointe en force brute.
    La force de couplage varie de 0.1 à 3.0.
    """
    couplings = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    chain_length = 5
    results = []

    for J in couplings:
        bp_errs, trw_errs, mf_errs = [], [], []

        for run in range(n_runs):
            fg = generate_chain(chain_length, cardinality=2, coupling=J, seed=run)

            # Marginales exactes par force brute
            joint = fg.joint_distribution()
            exact = {}
            for var in fg.variables:
                marg = joint
                for other in fg.variables:
                    if other != var:
                        marg = marg.marginalize(other)
                marg.normalize()
                exact[var] = marg.values

            # BP
            beliefs_bp, _, _ = sum_product_bp(fg, max_iter=200)
            bp_err = np.mean([
                np.max(np.abs(beliefs_bp[v] - exact[v]))
                for v in fg.variables
            ])
            bp_errs.append(bp_err)

            # TRW-S
            beliefs_trw, _, _ = trw_s(fg, max_iter=200, damping=0.0)
            trw_err = np.mean([
                np.max(np.abs(beliefs_trw[v] - exact[v]))
                for v in fg.variables
            ])
            trw_errs.append(trw_err)

            # Champ moyen
            q_mf, _, _ = mean_field_vi(fg, max_iter=200, seed=run)
            mf_err = np.mean([
                np.max(np.abs(q_mf[v] - exact[v]))
                for v in fg.variables
            ])
            mf_errs.append(mf_err)

        results.append({
            "coupling": J,
            "bp_max_error": np.mean(bp_errs),
            "trw_max_error": np.mean(trw_errs),
            "mf_max_error": np.mean(mf_errs),
        })

    return results


def stereo_benchmark():
    """Exécute le benchmark de correspondance stéréo sur les jeux de données Middlebury.

    Retourne les métriques pour chaque méthode et jeu de données.
    """
    results = []
    for name in ["venus"]:
        _, _, method_results = run_stereo_experiment(
            name, methods=["bp", "trws", "mf"],
            max_iter=30, window_size=7,
            pairwise_weight=20.0, pairwise_trunc=2.0
        )
        for method, r in method_results.items():
            results.append({
                "dataset": name,
                "method": method,
                "energy": r["energy"],
                "bad_pixel_rate": r["bad_pixel_rate"],
                "mae": r["mae"],
                "rmse": r["rmse"],
                "time_s": r["time_s"],
            })
    return results


if __name__ == "__main__":
    os.makedirs("outputs/csv", exist_ok=True)

    print("Exécution de convergence vs taille du graphe...")
    conv_results = convergence_vs_graph_size()
    with open("outputs/csv/convergence_vs_size.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=conv_results[0].keys())
        writer.writeheader()
        writer.writerows(conv_results)
    print(f"  {len(conv_results)} lignes enregistrées dans outputs/csv/convergence_vs_size.csv")

    print("Exécution de précision vs couplage...")
    acc_results = accuracy_vs_coupling()
    with open("outputs/csv/accuracy_vs_coupling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=acc_results[0].keys())
        writer.writeheader()
        writer.writerows(acc_results)
    print(f"  {len(acc_results)} lignes enregistrées dans outputs/csv/accuracy_vs_coupling.csv")

    print("Exécution du benchmark stéréo...")
    stereo_results = stereo_benchmark()
    with open("outputs/csv/stereo_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stereo_results[0].keys())
        writer.writeheader()
        writer.writerows(stereo_results)
    print(f"  {len(stereo_results)} lignes enregistrées dans outputs/csv/stereo_benchmark.csv")

    print("Terminé.")
