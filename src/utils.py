"""
Fonctions utilitaires pour la visualisation, la génération de données et l'évaluation.

Auteur : Mouhssine Rifaki
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .factor_graph import Factor, FactorGraph


def generate_ising_grid(rows, cols, coupling=1.0, field=0.0, seed=None):
    """Génère un modèle d'Ising 2D sous forme de graphe factoriel sur une grille.

    Le modèle d'Ising a des variables binaires (états 0 et 1) sur une grille,
    avec des potentiels par paires encourageant les variables voisines à s'accorder
    et des potentiels unaires optionnels (champ externe).

    Potentiel par paires :
        f(x_i, x_j) = exp(coupling * I(x_i == x_j))

    Potentiel unaire :
        f(x_i) = exp(field * (2*x_i - 1))  (biais vers l'état 1 si field > 0)

    Paramètres
    ----------
    rows, cols : int
        Dimensions de la grille.
    coupling : float
        Force de couplage. Les valeurs positives encouragent l'accord.
    field : float
        Force du champ externe. 0 signifie pas de biais.
    seed : int ou None
        Si fourni, ajoute une perturbation aléatoire au champ à chaque site.

    Retourne
    -------
    fg : FactorGraph
        Le graphe factoriel d'Ising.
    grid_vars : np.ndarray de forme (rows, cols), dtype=str
        Nom de la variable à chaque position de la grille.
    """
    rng = np.random.default_rng(seed)
    fg = FactorGraph(name=f"Ising_{rows}x{cols}")
    grid_vars = np.empty((rows, cols), dtype=object)

    # Ajout des variables
    for i in range(rows):
        for j in range(cols):
            name = f"x_{i}_{j}"
            grid_vars[i, j] = name
            fg.add_variable(name, 2)

    # Potentiel de couplage par paires
    pair_vals = np.array([
        [np.exp(coupling), np.exp(-coupling)],
        [np.exp(-coupling), np.exp(coupling)],
    ])

    # Ajout des facteurs par paires (arêtes horizontales et verticales)
    for i in range(rows):
        for j in range(cols):
            # Voisin de droite
            if j + 1 < cols:
                f = Factor(
                    [grid_vars[i, j], grid_vars[i, j + 1]],
                    [2, 2],
                    pair_vals.copy(),
                    name=f"edge_{i}_{j}_h",
                )
                fg.add_factor(f)
            # Voisin du bas
            if i + 1 < rows:
                f = Factor(
                    [grid_vars[i, j], grid_vars[i + 1, j]],
                    [2, 2],
                    pair_vals.copy(),
                    name=f"edge_{i}_{j}_v",
                )
                fg.add_factor(f)

    # Facteurs unaires (champ externe)
    for i in range(rows):
        for j in range(cols):
            local_field = field
            if seed is not None:
                local_field += 0.2 * rng.standard_normal()
            unary = np.array([np.exp(-local_field), np.exp(local_field)])
            f = Factor(
                [grid_vars[i, j]], [2], unary, name=f"field_{i}_{j}"
            )
            fg.add_factor(f)

    return fg, grid_vars


def generate_random_fg(n_vars, n_factors, max_cardinality=3, max_arity=3, seed=None):
    """Génère un graphe factoriel aléatoire pour les tests.

    Paramètres
    ----------
    n_vars : int
        Nombre de variables.
    n_factors : int
        Nombre de facteurs.
    max_cardinality : int
        Nombre maximal d'états par variable.
    max_arity : int
        Nombre maximal de variables par facteur.
    seed : int ou None
        Graine aléatoire.

    Retourne
    -------
    fg : FactorGraph
    """
    rng = np.random.default_rng(seed)
    fg = FactorGraph(name="RandomFG")

    var_names = [f"v{i}" for i in range(n_vars)]
    cardinalities = rng.integers(2, max_cardinality + 1, size=n_vars)

    for name, card in zip(var_names, cardinalities):
        fg.add_variable(name, int(card))

    for fi in range(n_factors):
        arity = rng.integers(1, min(max_arity + 1, n_vars + 1))
        chosen = rng.choice(n_vars, size=arity, replace=False)
        fvars = [var_names[c] for c in chosen]
        fcards = [int(cardinalities[c]) for c in chosen]
        vals = rng.exponential(1.0, size=tuple(fcards))
        f = Factor(fvars, fcards, vals, name=f"f{fi}")
        fg.add_factor(f)

    return fg


def generate_chain(length, cardinality=2, coupling=1.0, seed=None):
    """Génère un graphe factoriel en chaîne (arbre, sans cycles).

    Paramètres
    ----------
    length : int
        Nombre de variables dans la chaîne.
    cardinality : int
        Nombre d'états par variable.
    coupling : float
        Force de couplage pour les facteurs par paires.
    seed : int ou None
        Graine aléatoire pour les potentiels unaires.

    Retourne
    -------
    fg : FactorGraph
    """
    rng = np.random.default_rng(seed)
    fg = FactorGraph(name=f"Chain_{length}")

    var_names = [f"c{i}" for i in range(length)]
    for v in var_names:
        fg.add_variable(v, cardinality)

    # Potentiels par paires
    pair_vals = np.exp(coupling * np.eye(cardinality))
    for i in range(length - 1):
        f = Factor(
            [var_names[i], var_names[i + 1]],
            [cardinality, cardinality],
            pair_vals.copy(),
            name=f"pair_{i}_{i+1}",
        )
        fg.add_factor(f)

    # Potentiels unaires (aléatoires)
    for i in range(length):
        unary = rng.exponential(1.0, size=cardinality)
        f = Factor([var_names[i]], [cardinality], unary, name=f"unary_{i}")
        fg.add_factor(f)

    return fg


def plot_beliefs(beliefs, var_order=None, figsize=(12, 4)):
    """Trace les croyances marginales pour chaque variable sous forme de diagramme en barres.

    Paramètres
    ----------
    beliefs : dict
        var_name -> np.ndarray de forme (cardinalité,).
    var_order : list of str ou None
        Ordre d'affichage des variables. Si None, tri alphabétique.
    figsize : tuple
        Taille de la figure.

    Retourne
    -------
    fig : matplotlib.figure.Figure
    """
    if var_order is None:
        var_order = sorted(beliefs.keys())

    n = len(var_order)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, var in zip(axes, var_order):
        b = beliefs[var]
        states = np.arange(len(b))
        ax.bar(states, b, color="#3498db", edgecolor="#2c3e50", alpha=0.8)
        ax.set_title(var, fontsize=9)
        ax.set_xlabel("État")
        ax.set_ylabel("P")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(states)

    plt.tight_layout()
    return fig


def plot_convergence(history, title="Convergence de la BP", figsize=(8, 4)):
    """Trace la courbe de convergence (changement maximal de message par itération).

    Paramètres
    ----------
    history : list of float
        Changement maximal de message à chaque itération.
    title : str
        Titre du graphique.
    figsize : tuple
        Taille de la figure.

    Retourne
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(history) + 1), history, "b-", lw=1.5)
    ax.set_xlabel("Itération")
    ax.set_ylabel("Changement maximal de message")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_ising_beliefs(beliefs, grid_vars, figsize=(6, 5)):
    """Visualise les croyances marginales sur une grille d'Ising 2D sous forme de carte de chaleur.

    Paramètres
    ----------
    beliefs : dict
        var_name -> np.ndarray de forme (2,).
    grid_vars : np.ndarray de forme (rows, cols), dtype=str
        Nom de la variable à chaque position de la grille.
    figsize : tuple
        Taille de la figure.

    Retourne
    -------
    fig : matplotlib.figure.Figure
    """
    rows, cols = grid_vars.shape
    p1 = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            var = grid_vars[i, j]
            if var in beliefs:
                p1[i, j] = beliefs[var][1]  # P(x = 1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(p1, cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("P(x = 1) pour chaque variable de la grille")
    plt.colorbar(im, ax=ax, label="P(x=1)")
    ax.set_xlabel("Colonne")
    ax.set_ylabel("Ligne")
    plt.tight_layout()
    return fig


def plot_stereo_results(left, gt_disparity, results, dataset_name,
                        figsize=(16, 4)):
    """Affiche les résultats de correspondance stéréo côte à côte.

    Paramètres
    ----------
    left : np.ndarray (H, W)
    gt_disparity : np.ndarray (H, W)
    results : dict
        méthode -> {labeling, bad_pixel_rate, ...}
    dataset_name : str
    figsize : tuple

    Retourne
    -------
    fig : matplotlib.figure.Figure
    """
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods + 2, figsize=figsize)

    axes[0].imshow(left, cmap="gray")
    axes[0].set_title("Image gauche")
    axes[0].axis("off")

    vmax = gt_disparity.max()
    axes[1].imshow(gt_disparity, cmap="jet", vmin=0, vmax=vmax)
    axes[1].set_title("Vérité terrain")
    axes[1].axis("off")

    for ax, (method, res) in zip(axes[2:], results.items()):
        ax.imshow(res["labeling"], cmap="jet", vmin=0, vmax=vmax)
        bpr = res["bad_pixel_rate"]
        ax.set_title(f"{method.upper()}\n{bpr:.1%} mauvais")
        ax.axis("off")

    fig.suptitle(f"Correspondance stéréo : {dataset_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_energy_comparison(results, dataset_name, figsize=(10, 5)):
    """Trace les courbes de convergence de l'énergie pour toutes les méthodes.

    Paramètres
    ----------
    results : dict
        méthode -> {energy_history, [bound_history]}
    dataset_name : str
    figsize : tuple

    Retourne
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {"bp": "#3498db", "trws": "#e74c3c", "mf": "#2ecc71"}

    for method, res in results.items():
        if "energy_history" in res and len(res["energy_history"]) > 0:
            ax.plot(res["energy_history"], label=f"{method.upper()} énergie",
                    color=colors.get(method, "gray"), linewidth=1.5)
        if "bound_history" in res and len(res["bound_history"]) > 0:
            ax.plot(res["bound_history"], label=f"{method.upper()} borne",
                    color=colors.get(method, "gray"), linewidth=1.5,
                    linestyle="--")

    ax.set_xlabel("Itération")
    ax.set_ylabel("Énergie")
    ax.set_title(f"Convergence de l'énergie : {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
