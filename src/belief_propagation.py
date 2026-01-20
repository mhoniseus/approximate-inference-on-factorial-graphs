"""
Algorithmes de propagation de croyances pour les graphes factoriels.

Implémente l'algorithme somme-produit (marginales exactes sur les arbres), max-produit
(MAP sur les arbres), et la BP bouclée (marginales approchées sur les graphes avec cycles).

Auteur : Mouhssine Rifaki
"""

import numpy as np
from .factor_graph import FactorGraph


def _init_messages(fg):
    """Initialise tous les messages avec des distributions uniformes.

    Retourne deux dictionnaires :
        msg_var_to_fac[(var, fac_idx)] = tableau de forme (cardinalité,)
        msg_fac_to_var[(fac_idx, var)] = tableau de forme (cardinalité,)
    """
    msg_var_to_fac = {}
    msg_fac_to_var = {}

    for var_name, card in fg.variables.items():
        for fi in fg.neighbors_of_variable(var_name):
            msg_var_to_fac[(var_name, fi)] = np.ones(card) / card
            msg_fac_to_var[(fi, var_name)] = np.ones(card) / card

    return msg_var_to_fac, msg_fac_to_var


def _update_var_to_fac(fg, var_name, target_fi, msg_fac_to_var):
    """Calcule le message de la variable `var_name` vers le facteur `target_fi`.

    Le message est le produit de tous les messages facteur-vers-variable
    entrants sauf celui provenant de `target_fi`.
    """
    card = fg.variables[var_name]
    msg = np.ones(card)
    for fi in fg.neighbors_of_variable(var_name):
        if fi != target_fi:
            msg *= msg_fac_to_var[(fi, var_name)]
    # Normalisation pour la stabilité numérique
    total = msg.sum()
    if total > 0:
        msg /= total
    return msg


def _update_fac_to_var(fg, fac_idx, target_var, msg_var_to_fac, mode="sum"):
    """Calcule le message du facteur `fac_idx` vers la variable `target_var`.

    Paramètres
    ----------
    mode : str
        "sum" pour somme-produit (marginales), "max" pour max-produit (MAP).
    """
    factor = fg.factors[fac_idx]
    other_vars = [v for v in factor.variables if v != target_var]

    # Commence avec la table de potentiel du facteur
    # On doit multiplier les messages de toutes les autres variables,
    # puis marginaliser (somme ou max) sur ces variables.

    # Construire le produit des valeurs du facteur et des messages entrants
    combined = factor.values.copy()

    for v in other_vars:
        incoming = msg_var_to_fac[(v, fac_idx)]
        # Redimensionner pour diffuser le long de l'axe correct
        axis = factor.variables.index(v)
        shape = [1] * len(factor.variables)
        shape[axis] = len(incoming)
        combined = combined * incoming.reshape(shape)

    # Marginaliser sur toutes les variables sauf target_var
    target_axis = factor.variables.index(target_var)
    axes_to_reduce = [
        factor.variables.index(v) for v in other_vars
    ]
    # Trier en ordre inverse pour éviter le décalage des axes
    for ax in sorted(axes_to_reduce, reverse=True):
        if mode == "sum":
            combined = combined.sum(axis=ax)
        else:
            combined = combined.max(axis=ax)

    msg = combined.ravel()

    # Normalisation
    total = msg.sum()
    if total > 0:
        msg /= total
    return msg


def _compute_beliefs(fg, msg_fac_to_var):
    """Calcule les croyances marginales pour chaque variable à partir des messages facteur-vers-variable.

    Retourne
    -------
    beliefs : dict
        var_name -> np.ndarray de forme (cardinalité,), normalisé.
    """
    beliefs = {}
    for var_name, card in fg.variables.items():
        b = np.ones(card)
        for fi in fg.neighbors_of_variable(var_name):
            b *= msg_fac_to_var[(fi, var_name)]
        total = b.sum()
        if total > 0:
            b /= total
        beliefs[var_name] = b
    return beliefs


def sum_product_bp(fg, max_iter=100):
    """Propagation de croyances somme-produit sur un graphe factoriel en arbre.

    Calcule les distributions marginales exactes pour chaque variable en exécutant
    deux passes (feuilles vers racine, racine vers feuilles) sur un arbre.

    Pour les arbres, la convergence est garantie en au plus `diamètre` itérations.
    Sur les graphes bouclés, cela se réduit à la BP bouclée standard.

    Paramètres
    ----------
    fg : FactorGraph
        Le graphe factoriel (doit être un arbre pour des résultats exacts).
    max_iter : int
        Nombre maximal d'itérations de passage de messages.

    Retourne
    -------
    beliefs : dict
        var_name -> np.ndarray de forme (cardinalité,).
    converged : bool
        Indique si les messages ont convergé avant max_iter.
    n_iter : int
        Nombre d'itérations effectivement réalisées.
    """
    msg_v2f, msg_f2v = _init_messages(fg)
    converged = False

    for it in range(max_iter):
        old_f2v = {k: v.copy() for k, v in msg_f2v.items()}

        # Mise à jour des messages variable-vers-facteur
        for var_name in fg.variables:
            for fi in fg.neighbors_of_variable(var_name):
                msg_v2f[(var_name, fi)] = _update_var_to_fac(
                    fg, var_name, fi, msg_f2v
                )

        # Mise à jour des messages facteur-vers-variable
        for fi in range(fg.n_factors):
            for var_name in fg.neighbors_of_factor(fi):
                msg_f2v[(fi, var_name)] = _update_fac_to_var(
                    fg, fi, var_name, msg_v2f, mode="sum"
                )

        # Vérification de la convergence
        max_diff = 0.0
        for key in msg_f2v:
            max_diff = max(max_diff, np.max(np.abs(msg_f2v[key] - old_f2v[key])))
        if max_diff < 1e-10:
            converged = True
            break

    beliefs = _compute_beliefs(fg, msg_f2v)
    return beliefs, converged, it + 1


def max_product_bp(fg, max_iter=100):
    """Propagation de croyances max-produit pour l'estimation MAP.

    Remplace la sommation par la maximisation dans les mises à jour de messages.
    Sur un arbre, cela donne l'assignation MAP exacte.

    Paramètres
    ----------
    fg : FactorGraph
        Le graphe factoriel (doit être un arbre pour un MAP exact).
    max_iter : int
        Nombre maximal d'itérations de passage de messages.

    Retourne
    -------
    map_assignment : dict
        var_name -> int (indice de l'état avec la croyance la plus élevée).
    beliefs : dict
        var_name -> np.ndarray de forme (cardinalité,).
    converged : bool
        Indique si les messages ont convergé avant max_iter.
    n_iter : int
        Nombre d'itérations effectivement réalisées.
    """
    msg_v2f, msg_f2v = _init_messages(fg)
    converged = False

    for it in range(max_iter):
        old_f2v = {k: v.copy() for k, v in msg_f2v.items()}

        for var_name in fg.variables:
            for fi in fg.neighbors_of_variable(var_name):
                msg_v2f[(var_name, fi)] = _update_var_to_fac(
                    fg, var_name, fi, msg_f2v
                )

        for fi in range(fg.n_factors):
            for var_name in fg.neighbors_of_factor(fi):
                msg_f2v[(fi, var_name)] = _update_fac_to_var(
                    fg, fi, var_name, msg_v2f, mode="max"
                )

        max_diff = 0.0
        for key in msg_f2v:
            max_diff = max(max_diff, np.max(np.abs(msg_f2v[key] - old_f2v[key])))
        if max_diff < 1e-10:
            converged = True
            break

    beliefs = _compute_beliefs(fg, msg_f2v)
    map_assignment = {v: int(np.argmax(b)) for v, b in beliefs.items()}
    return map_assignment, beliefs, converged, it + 1


def loopy_bp(fg, max_iter=200, damping=0.5, tol=1e-8):
    """Propagation de croyances bouclée avec amortissement.

    Sur les graphes avec cycles, la BP standard peut ne pas converger. L'amortissement
    ralentit les mises à jour des messages pour améliorer la stabilité :

        m_new = damping * m_old + (1 - damping) * m_update

    Paramètres
    ----------
    fg : FactorGraph
        Le graphe factoriel (peut contenir des cycles).
    max_iter : int
        Nombre maximal d'itérations.
    damping : float
        Coefficient d'amortissement dans [0, 1). Des valeurs plus élevées signifient
        des mises à jour plus lentes.
    tol : float
        Seuil de convergence sur le changement maximal de message.

    Retourne
    -------
    beliefs : dict
        var_name -> np.ndarray de forme (cardinalité,).
    converged : bool
        Indique si les messages ont convergé avant max_iter.
    history : list of float
        Changement maximal de message à chaque itération (pour le suivi).
    """
    assert 0.0 <= damping < 1.0

    msg_v2f, msg_f2v = _init_messages(fg)
    history = []

    converged = False
    for it in range(max_iter):
        old_f2v = {k: v.copy() for k, v in msg_f2v.items()}

        # Variable-vers-facteur
        for var_name in fg.variables:
            for fi in fg.neighbors_of_variable(var_name):
                new_msg = _update_var_to_fac(fg, var_name, fi, msg_f2v)
                msg_v2f[(var_name, fi)] = (
                    damping * msg_v2f[(var_name, fi)] + (1 - damping) * new_msg
                )

        # Facteur-vers-variable
        for fi in range(fg.n_factors):
            for var_name in fg.neighbors_of_factor(fi):
                new_msg = _update_fac_to_var(fg, fi, var_name, msg_v2f, mode="sum")
                msg_f2v[(fi, var_name)] = (
                    damping * msg_f2v[(fi, var_name)] + (1 - damping) * new_msg
                )

        max_diff = 0.0
        for key in msg_f2v:
            max_diff = max(max_diff, np.max(np.abs(msg_f2v[key] - old_f2v[key])))
        history.append(max_diff)

        if max_diff < tol:
            converged = True
            break

    beliefs = _compute_beliefs(fg, msg_f2v)
    return beliefs, converged, history
