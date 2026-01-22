"""
Passage de messages séquentiel pondéré par arbres (TRW-S) pour les graphes factoriels.

Implémente TRW-S (Kolmogorov 2006) qui fournit une borne supérieure sur log Z
et offre de meilleures propriétés de convergence que la BP bouclée.
"""

import numpy as np
from scipy.optimize import linprog
from .factor_graph import FactorGraph


def _log_safe(x, eps=1e-30):
    """Logarithme sûr pour éviter log(0)."""
    return np.log(np.maximum(x, eps))


def _compute_rho(fg):
    """Calcule les probabilités d'apparition dans les arbres couvrants (rho) pour chaque facteur.

    Pour les facteurs unaires : rho = 1.0
    Pour les facteurs par paires :
      - Si le graphe est un arbre : rho = 1.0 (TRW se réduit à BP)
      - Sinon : rho = 0.5 (standard pour les grilles)

    Retourne
    -------
    rho : dict
        indice_facteur -> float
    """
    is_tree = fg.is_tree()
    rho = {}
    for fi, factor in enumerate(fg.factors):
        if len(factor.variables) == 1:
            rho[fi] = 1.0
        else:
            rho[fi] = 1.0 if is_tree else 0.5
    return rho


def compute_mrf_energy(fg, assignment):
    """Calcule l'énergie MRF E(x) = -sum_a log f_a(x_a) pour une assignation donnée.

    Paramètres
    ----------
    fg : FactorGraph
    assignment : dict
        var_name -> int (indice d'état)

    Retourne
    -------
    energy : float
    """
    energy = 0.0
    for factor in fg.factors:
        idx = tuple(assignment[v] for v in factor.variables)
        energy -= _log_safe(factor.values[idx])
    return energy


def _trw_update_fac_to_var(fg, fac_idx, target_var, msg_var_to_fac, rho):
    """Message facteur-vers-variable pour TRW.

    mu_{a->i}(x_i) propto sum_{x_a\\i} f_a(x_a)^{1/rho_a} * prod_{j!=i} mu_{j->a}(x_j)
    """
    factor = fg.factors[fac_idx]
    rho_a = rho[fac_idx]
    other_vars = [v for v in factor.variables if v != target_var]

    # Potentiel élevé à la puissance 1/rho
    combined = np.power(np.maximum(factor.values, 1e-30), 1.0 / rho_a)

    for v in other_vars:
        incoming = msg_var_to_fac[(v, fac_idx)]
        axis = factor.variables.index(v)
        shape = [1] * len(factor.variables)
        shape[axis] = len(incoming)
        combined = combined * incoming.reshape(shape)

    # Marginalisation sur toutes les variables sauf target_var
    target_axis = factor.variables.index(target_var)
    axes_to_reduce = [factor.variables.index(v) for v in other_vars]
    for ax in sorted(axes_to_reduce, reverse=True):
        combined = combined.sum(axis=ax)

    msg = combined.ravel()
    total = msg.sum()
    if total > 0:
        msg /= total
    return msg


def _compute_trw_beliefs(fg, msg_fac_to_var, rho):
    """Calcule les croyances TRW : b_i(x_i) propto prod_a mu_{a->i}(x_i)^{rho_a}."""
    beliefs = {}
    for var_name, card in fg.variables.items():
        log_b = np.zeros(card)
        for fi in fg.neighbors_of_variable(var_name):
            log_b += rho[fi] * _log_safe(msg_fac_to_var[(fi, var_name)])
        log_b -= np.max(log_b)
        b = np.exp(log_b)
        total = b.sum()
        if total > 0:
            b /= total
        beliefs[var_name] = b
    return beliefs


def _compute_trw_bound_from_messages(fg, msg_fac_to_var, msg_var_to_fac, rho):
    """Calcule une approximation de la borne TRW à partir des messages courants.

    Utilisé pour suivre la convergence (valeur indicative, pas une borne valide
    en cours d'itération).
    """
    beliefs = _compute_trw_beliefs(fg, msg_fac_to_var, rho)

    bound = 0.0

    for fi, factor in enumerate(fg.factors):
        rho_a = rho[fi]
        is_unary = len(factor.variables) == 1

        b_a_unnorm = np.power(np.maximum(factor.values, 1e-30), 1.0 / rho_a)
        for v in factor.variables:
            incoming = msg_var_to_fac[(v, fi)]
            axis = factor.variables.index(v)
            shape = [1] * len(factor.variables)
            shape[axis] = len(incoming)
            b_a_unnorm = b_a_unnorm * incoming.reshape(shape)

        Z_a = b_a_unnorm.sum()
        b_a = b_a_unnorm / Z_a if Z_a > 0 else b_a_unnorm

        log_f = _log_safe(factor.values)
        bound += np.sum(b_a * log_f)

        if not is_unary:
            bound -= rho_a * np.sum(b_a * _log_safe(b_a))

    for var_name in fg.variables:
        kappa = sum(rho[fi] for fi in fg.neighbors_of_variable(var_name)
                    if len(fg.factors[fi].variables) > 1)
        b = beliefs[var_name]
        H = -np.sum(b * _log_safe(b))
        bound += (1.0 - kappa) * H

    return bound


def compute_trw_bound(fg, rho=None):
    """Calcule la borne supérieure TRW sur log Z avec les marginales exactes.

    Utilise le calcul en force brute des marginales unaires et par paires,
    puis évalue la borne TRW (Wainwright et al. 2005) :

        log Z ≤ Σ_i E_{μ_i}[log f_i] + Σ_{ij} E_{μ_{ij}}[log f_{ij}]
              + Σ_{ij} ρ_{ij} H(μ_{ij}) + Σ_i (1 - κ_i) H(μ_i)

    Ne convient qu'aux petits graphes (calcul en force brute).

    Paramètres
    ----------
    fg : FactorGraph
    rho : dict ou None
        Probabilités d'apparition par facteur. Si None, calculées automatiquement.

    Retourne
    -------
    bound : float
        Borne supérieure sur log Z.
    """
    from itertools import product as cartesian_product

    if rho is None:
        rho = _compute_rho(fg)

    all_vars = sorted(fg.variables.keys())
    cards = {v: fg.variables[v] for v in all_vars}

    # Calcul de la distribution jointe non normalisée
    configs = list(cartesian_product(*[range(cards[v]) for v in all_vars]))
    joint_unnorm = np.zeros(len(configs))
    for ci, states in enumerate(configs):
        assign = dict(zip(all_vars, states))
        val = 1.0
        for f in fg.factors:
            idx = tuple(assign[v] for v in f.variables)
            val *= f.values[idx]
        joint_unnorm[ci] = val

    # Marginales exactes unaires
    unary_marginals = {}
    for vi, v in enumerate(all_vars):
        marg = np.zeros(cards[v])
        for ci, states in enumerate(configs):
            marg[states[vi]] += joint_unnorm[ci]
        marg /= marg.sum()
        unary_marginals[v] = marg

    # Marginales exactes par paires (pour les facteurs pairwise)
    pair_marginals = {}
    for fi, factor in enumerate(fg.factors):
        if len(factor.variables) != 2:
            continue
        vi_name, vj_name = factor.variables
        vi_idx = all_vars.index(vi_name)
        vj_idx = all_vars.index(vj_name)
        Ki, Kj = cards[vi_name], cards[vj_name]
        marg = np.zeros((Ki, Kj))
        for ci, states in enumerate(configs):
            marg[states[vi_idx], states[vj_idx]] += joint_unnorm[ci]
        marg /= marg.sum()
        pair_marginals[fi] = marg

    # Évaluation de la borne TRW
    bound = 0.0

    for fi, factor in enumerate(fg.factors):
        log_f = _log_safe(factor.values)
        is_unary = len(factor.variables) == 1

        if is_unary:
            v = factor.variables[0]
            bound += np.sum(unary_marginals[v] * log_f.ravel())
        else:
            b_pair = pair_marginals[fi]
            bound += np.sum(b_pair * log_f)
            bound -= rho[fi] * np.sum(b_pair * _log_safe(b_pair))

    for var_name in all_vars:
        kappa = sum(rho[fi] for fi in fg.neighbors_of_variable(var_name)
                    if len(fg.factors[fi].variables) > 1)
        b = unary_marginals[var_name]
        H = -np.sum(b * _log_safe(b))
        bound += (1.0 - kappa) * H

    return bound


def trw_s(fg, max_iter=200, damping=0.5, tol=1e-8):
    """Passage de messages séquentiel pondéré par arbres (TRW-S).

    Implémente la variante séquentielle de TRW (Kolmogorov 2006).
    Les messages sont mis à jour dans un ordre fixe (balayages avant et arrière),
    ce qui améliore la convergence par rapport à la BP bouclée standard.

    Paramètres
    ----------
    fg : FactorGraph
        Le graphe factoriel (peut contenir des cycles).
    max_iter : int
        Nombre maximal de balayages avant+arrière.
    damping : float
        Coefficient d'amortissement dans [0, 1).
    tol : float
        Seuil de convergence sur le changement maximal de message.

    Retourne
    -------
    beliefs : dict
        var_name -> np.ndarray de forme (cardinalité,).
    converged : bool
    bound_history : list of float
        Borne supérieure TRW à chaque itération.
    """
    assert 0.0 <= damping < 1.0

    rho = _compute_rho(fg)

    # Initialisation des messages
    msg_v2f = {}
    msg_f2v = {}
    for var_name, card in fg.variables.items():
        for fi in fg.neighbors_of_variable(var_name):
            msg_v2f[(var_name, fi)] = np.ones(card) / card
            msg_f2v[(fi, var_name)] = np.ones(card) / card

    bound_history = []
    converged = False

    var_order = sorted(fg.variables.keys())

    for it in range(max_iter):
        old_f2v = {k: v.copy() for k, v in msg_f2v.items()}

        # Balayage avant : traitement séquentiel des variables
        for var_name in var_order:
            _update_variable_messages(
                fg, var_name, msg_v2f, msg_f2v, rho, damping
            )

        # Balayage arrière
        for var_name in reversed(var_order):
            _update_variable_messages(
                fg, var_name, msg_v2f, msg_f2v, rho, damping
            )

        # Calcul de la borne (indicatif, pas une borne valide en cours d'itération)
        bound = _compute_trw_bound_from_messages(fg, msg_f2v, msg_v2f, rho)
        bound_history.append(bound)

        # Vérification de la convergence
        max_diff = 0.0
        for key in msg_f2v:
            max_diff = max(max_diff, np.max(np.abs(msg_f2v[key] - old_f2v[key])))

        if max_diff < tol:
            converged = True
            break

    beliefs = _compute_trw_beliefs(fg, msg_f2v, rho)
    return beliefs, converged, bound_history


def _update_variable_messages(fg, var_name, msg_v2f, msg_f2v, rho, damping):
    """Met à jour tous les messages liés à une variable (séquentiel).

    1. Met à jour fac->var pour tous les facteurs contenant var_name
    2. Met à jour var->fac pour tous les facteurs contenant var_name
    """
    card = fg.variables[var_name]

    # Mise à jour des messages facteur-vers-variable vers cette variable
    for fi in fg.neighbors_of_variable(var_name):
        new_msg = _trw_update_fac_to_var(fg, fi, var_name, msg_v2f, rho)
        msg_f2v[(fi, var_name)] = (
            damping * msg_f2v[(fi, var_name)] + (1 - damping) * new_msg
        )

    # Mise à jour des messages variable-vers-facteur depuis cette variable
    for fi in fg.neighbors_of_variable(var_name):
        msg = np.ones(card)
        for fj in fg.neighbors_of_variable(var_name):
            if fj != fi:
                msg *= msg_f2v[(fj, var_name)]
        total = msg.sum()
        if total > 0:
            msg /= total
        msg_v2f[(var_name, fi)] = msg


def lp_relaxation_map(fg):
    """Résout la relaxation LP du problème MAP sur un graphe factoriel.

    Formule la relaxation LP sur le polytope marginal local :

        min_{mu} sum_a sum_{x_a} theta_a(x_a) * mu_a(x_a)
        s.t. mu_i(x_i) >= 0, sum_{x_i} mu_i(x_i) = 1
             marginalization consistency for pairwise factors

    La solution donne une borne inférieure sur l'énergie MAP exacte.

    Paramètres
    ----------
    fg : FactorGraph
        Graphe factoriel (seuls les facteurs unaires et par paires sont supportés).

    Retourne
    -------
    lp_energy : float
        Borne inférieure sur l'énergie MAP (relaxation LP).
    mu_vars : dict
        var_name -> np.ndarray, pseudo-marginales optimales.
    """
    var_list = sorted(fg.variables.keys())
    var_cards = {v: fg.variables[v] for v in var_list}

    # Identifier facteurs unaires et par paires
    unary_factors = []
    pairwise_factors = []
    for fi, factor in enumerate(fg.factors):
        if len(factor.variables) == 1:
            unary_factors.append(fi)
        elif len(factor.variables) == 2:
            pairwise_factors.append(fi)
        else:
            raise ValueError("LP relaxation ne supporte que les facteurs unaires et par paires.")

    # Construction des indices dans le vecteur LP
    # Variables : [mu_i(x_i) pour chaque i, x_i] + [mu_ij(x_i,x_j) pour chaque paire]
    offset = {}
    idx = 0
    for v in var_list:
        offset[v] = idx
        idx += var_cards[v]
    n_unary_vars = idx

    pair_offset = {}
    for fi in pairwise_factors:
        factor = fg.factors[fi]
        pair_offset[fi] = idx
        idx += factor.cardinalities[0] * factor.cardinalities[1]
    n_total = idx

    # Vecteur de coût : c = -log f_a(x_a) pour chaque composante
    c = np.zeros(n_total)
    for fi in unary_factors:
        factor = fg.factors[fi]
        v = factor.variables[0]
        theta = -_log_safe(factor.values.ravel())
        c[offset[v]:offset[v] + var_cards[v]] += theta

    for fi in pairwise_factors:
        factor = fg.factors[fi]
        theta = -_log_safe(factor.values.ravel())
        c[pair_offset[fi]:pair_offset[fi] + len(theta)] += theta

    # Contraintes d'égalité
    A_eq_rows = []
    b_eq_rows = []

    # 1. Normalisation : sum_{x_i} mu_i(x_i) = 1 pour chaque variable
    for v in var_list:
        row = np.zeros(n_total)
        row[offset[v]:offset[v] + var_cards[v]] = 1.0
        A_eq_rows.append(row)
        b_eq_rows.append(1.0)

    # 2. Cohérence de marginalisation pour les facteurs par paires
    for fi in pairwise_factors:
        factor = fg.factors[fi]
        vi, vj = factor.variables[0], factor.variables[1]
        Ki, Kj = factor.cardinalities[0], factor.cardinalities[1]
        po = pair_offset[fi]

        # sum_{x_j} mu_{ij}(x_i, x_j) = mu_i(x_i) pour chaque x_i
        for xi in range(Ki):
            row = np.zeros(n_total)
            for xj in range(Kj):
                row[po + xi * Kj + xj] = 1.0
            row[offset[vi] + xi] = -1.0
            A_eq_rows.append(row)
            b_eq_rows.append(0.0)

        # sum_{x_i} mu_{ij}(x_i, x_j) = mu_j(x_j) pour chaque x_j
        for xj in range(Kj):
            row = np.zeros(n_total)
            for xi in range(Ki):
                row[po + xi * Kj + xj] = 1.0
            row[offset[vj] + xj] = -1.0
            A_eq_rows.append(row)
            b_eq_rows.append(0.0)

    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None

    # Bornes : toutes les variables >= 0
    bounds = [(0, None)] * n_total

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not result.success:
        raise RuntimeError(f"LP relaxation n'a pas convergé : {result.message}")

    # Extraire les pseudo-marginales
    mu_vars = {}
    for v in var_list:
        mu_vars[v] = result.x[offset[v]:offset[v] + var_cards[v]]

    return result.fun, mu_vars
