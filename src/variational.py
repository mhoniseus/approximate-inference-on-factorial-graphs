"""
Inférence variationnelle pour les modèles graphiques discrets.

Implémente l'approximation par champ moyen et le calcul de l'ELBO pour
les graphes factoriels avec des variables discrètes.

Auteur : Mouhssine Rifaki
"""

import numpy as np
from .factor_graph import FactorGraph


def _log_safe(x, eps=1e-30):
    """Logarithme avec un plancher minimal pour éviter log(0)."""
    return np.log(np.maximum(x, eps))


def compute_elbo(fg, q):
    """Calcule la borne inférieure de l'évidence (ELBO) sous une distribution à champ moyen.

    Sous l'approximation par champ moyen, q(x) = prod_i q_i(x_i).
    L'ELBO se décompose comme suit :

        ELBO = sum_f E_q[log f(x_f)] + sum_i H(q_i)

    où H(q_i) est l'entropie de la marginale variationnelle pour la variable i.

    Paramètres
    ----------
    fg : FactorGraph
        Le graphe factoriel définissant le modèle.
    q : dict
        var_name -> np.ndarray de forme (cardinalité,).
        Les marginales variationnelles (doivent chacune sommer à 1).

    Retourne
    -------
    float
        La valeur de l'ELBO.
    """
    elbo = 0.0

    # Termes d'entropie : sum_i H(q_i) = -sum_i sum_k q_i(k) log q_i(k)
    for var_name, qi in q.items():
        elbo -= np.sum(qi * _log_safe(qi))

    # Termes d'énergie : sum_f E_q[log f(x_f)]
    for factor in fg.factors:
        log_vals = _log_safe(factor.values)

        # Calcul de E_q[log f] en sommant sur toutes les configurations
        # pondérées par le produit des marginales
        n_vars = len(factor.variables)
        cards = factor.cardinalities

        expected = 0.0
        for idx in np.ndindex(*cards):
            weight = 1.0
            for axis, var in enumerate(factor.variables):
                weight *= q[var][idx[axis]]
            expected += weight * log_vals[idx]

        elbo += expected

    return elbo


def mean_field_vi(fg, max_iter=200, tol=1e-8, seed=None):
    """Inférence variationnelle par champ moyen avec montée par coordonnées.

    Met à jour itérativement chaque marginale variationnelle q_i pour maximiser l'ELBO
    en gardant toutes les autres marginales fixées. Chaque mise à jour a la forme :

        log q_i(x_i) = E_{q_{-i}}[log p(x)] + const

    ce qui pour un graphe factoriel devient :

        log q_i(x_i) = sum_{f : i in scope(f)} E_{q_{-i}}[log f(x_f)] + const

    Paramètres
    ----------
    fg : FactorGraph
        Le graphe factoriel définissant le modèle.
    max_iter : int
        Nombre maximal de balayages complets sur toutes les variables.
    tol : float
        Seuil de convergence sur le changement de l'ELBO.
    seed : int ou None
        Graine aléatoire pour l'initialisation des marginales variationnelles.

    Retourne
    -------
    q : dict
        var_name -> np.ndarray de forme (cardinalité,).
        Les marginales variationnelles optimisées.
    elbo_history : list of float
        Valeur de l'ELBO après chaque itération.
    converged : bool
        Indique si l'ELBO a convergé avant max_iter.
    """
    rng = np.random.default_rng(seed)

    # Initialisation de q avec des distributions tirées de Dirichlet
    q = {}
    for var_name, card in fg.variables.items():
        qi = rng.dirichlet(np.ones(card))
        q[var_name] = qi

    elbo_history = []
    converged = False

    for it in range(max_iter):
        # Montée par coordonnées : mise à jour de chaque q_i tour à tour
        for var_name in fg.variables:
            card = fg.variables[var_name]
            log_qi = np.zeros(card)

            # Somme des contributions de chaque facteur contenant var_name
            for fi in fg.neighbors_of_variable(var_name):
                factor = fg.factors[fi]
                log_vals = _log_safe(factor.values)
                other_vars = [v for v in factor.variables if v != var_name]
                target_axis = factor.variables.index(var_name)

                if len(other_vars) == 0:
                    # Facteur unaire : la contribution est simplement log f(x_i)
                    log_qi += log_vals.ravel()
                else:
                    # Calcul de E_{q_{-i}}[log f] pour chaque état de x_i
                    other_cards = [
                        factor.cardinalities[factor.variables.index(v)]
                        for v in other_vars
                    ]

                    for xi in range(card):
                        expected = 0.0
                        for other_idx in np.ndindex(*other_cards):
                            # Construire l'indice complet dans la table du facteur
                            full_idx = [0] * len(factor.variables)
                            full_idx[target_axis] = xi
                            for j, v in enumerate(other_vars):
                                axis = factor.variables.index(v)
                                full_idx[axis] = other_idx[j]

                            weight = 1.0
                            for j, v in enumerate(other_vars):
                                weight *= q[v][other_idx[j]]

                            expected += weight * log_vals[tuple(full_idx)]
                        log_qi[xi] += expected

            # Conversion en une distribution valide
            log_qi -= np.max(log_qi)  # stabilité numérique
            qi = np.exp(log_qi)
            qi /= qi.sum()
            q[var_name] = qi

        # Calcul de l'ELBO après le balayage complet
        elbo = compute_elbo(fg, q)
        elbo_history.append(elbo)

        if it > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            converged = True
            break

    return q, elbo_history, converged
