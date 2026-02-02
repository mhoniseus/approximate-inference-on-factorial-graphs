"""
Génération de graphiques de qualité publication à partir des résultats CSV de benchmark.

Lit depuis outputs/csv/ et écrit les graphiques PNG dans outputs/plots/.

Auteur : Mouhssine Rifaki
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def read_csv(path):
    """Lit un fichier CSV et retourne une liste de dictionnaires avec conversion en flottants."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_convergence_vs_size(rows, outdir):
    """Itérations et temps d'exécution vs taille du graphe pour BP, TRW-S et champ moyen."""
    sizes = [r["n_variables"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sizes, [r["bp_iters"] for r in rows], "o-", color="#3498db",
             label="BP bouclée", linewidth=2)
    ax1.plot(sizes, [r["trw_iters"] for r in rows], "^-", color="#2ecc71",
             label="TRW-S", linewidth=2)
    ax1.plot(sizes, [r["mf_iters"] for r in rows], "s--", color="#e74c3c",
             label="IV champ moyen", linewidth=2)
    ax1.set_xlabel("Nombre de variables")
    ax1.set_ylabel("Itérations pour converger")
    ax1.set_title("Vitesse de convergence vs taille du graphe")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(sizes, [r["bp_time_s"] for r in rows], "o-", color="#3498db",
             label="BP bouclée", linewidth=2)
    ax2.plot(sizes, [r["trw_time_s"] for r in rows], "^-", color="#2ecc71",
             label="TRW-S", linewidth=2)
    ax2.plot(sizes, [r["mf_time_s"] for r in rows], "s--", color="#e74c3c",
             label="IV champ moyen", linewidth=2)
    ax2.set_xlabel("Nombre de variables")
    ax2.set_ylabel("Temps (secondes)")
    ax2.set_title("Temps d'exécution vs taille du graphe")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "convergence_vs_size.png"), bbox_inches="tight")
    plt.close(fig)
    print("  convergence_vs_size.png enregistré")


def plot_accuracy_vs_coupling(rows, outdir):
    """Erreur marginale vs force de couplage pour BP, TRW-S et champ moyen."""
    couplings = [r["coupling"] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(couplings, [r["bp_max_error"] for r in rows], "o-", color="#3498db",
            label="BP somme-produit", linewidth=2)
    ax.plot(couplings, [r["trw_max_error"] for r in rows], "^-", color="#2ecc71",
            label="TRW-S", linewidth=2)
    ax.plot(couplings, [r["mf_max_error"] for r in rows], "s--", color="#e74c3c",
            label="IV champ moyen", linewidth=2)
    ax.set_xlabel("Force de couplage (J)")
    ax.set_ylabel("Erreur marginale maximale moyenne")
    ax.set_title("Qualité d'approximation vs force de couplage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_coupling.png"), bbox_inches="tight")
    plt.close(fig)
    print("  accuracy_vs_coupling.png enregistré")


def plot_stereo_benchmark(rows, outdir):
    """Métriques de correspondance stéréo pour chaque méthode."""
    methods = [r["method"] for r in rows]
    colors = {"bp": "#3498db", "trws": "#2ecc71", "mf": "#e74c3c"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Bad pixel rate
    for r in rows:
        axes[0].bar(r["method"].upper(), r["bad_pixel_rate"] * 100,
                     color=colors.get(r["method"], "#95a5a6"))
    axes[0].set_ylabel("Taux de mauvais pixels (%)")
    axes[0].set_title("Taux de mauvais pixels")
    axes[0].grid(True, alpha=0.3, axis='y')

    # MAE
    for r in rows:
        axes[1].bar(r["method"].upper(), r["mae"],
                     color=colors.get(r["method"], "#95a5a6"))
    axes[1].set_ylabel("Erreur absolue moyenne")
    axes[1].set_title("MAE")
    axes[1].grid(True, alpha=0.3, axis='y')

    # Energy
    for r in rows:
        axes[2].bar(r["method"].upper(), r["energy"],
                     color=colors.get(r["method"], "#95a5a6"))
    axes[2].set_ylabel("Énergie MRF")
    axes[2].set_title("Énergie finale")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Benchmark stéréo - Venus", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "stereo_benchmark.png"), bbox_inches="tight")
    plt.close(fig)
    print("  stereo_benchmark.png enregistré")


if __name__ == "__main__":
    outdir = "outputs/plots"
    os.makedirs(outdir, exist_ok=True)

    print("Génération des graphiques...")
    conv_rows = read_csv("outputs/csv/convergence_vs_size.csv")
    plot_convergence_vs_size(conv_rows, outdir)

    acc_rows = read_csv("outputs/csv/accuracy_vs_coupling.csv")
    plot_accuracy_vs_coupling(acc_rows, outdir)

    stereo_path = "outputs/csv/stereo_benchmark.csv"
    if os.path.exists(stereo_path):
        stereo_rows = read_csv(stereo_path)
        plot_stereo_benchmark(stereo_rows, outdir)

    print("Terminé.")
