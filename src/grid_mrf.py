"""
MRF efficace sur grille 2D avec inférence vectorisée.

Implémente une classe GridMRF spécialisée pour les problèmes à grande échelle
(comme la correspondance stéréo) où la représentation générique FactorGraph
serait trop lente. Les algorithmes d'inférence (BP, TRW-S, champ moyen)
opèrent directement sur des tableaux NumPy dans le domaine min-sum (énergie).
"""

import numpy as np
from .factor_graph import Factor, FactorGraph


class GridMRF:
    """MRF sur une grille 2D de pixels avec des étiquettes discrètes.

    Stocke les potentiels sous forme de tableaux NumPy denses et implémente
    le passage de messages vectorisé directement sur la structure de grille.

    Paramètres
    ----------
    height, width : int
        Dimensions de la grille.
    n_labels : int
        Nombre d'étiquettes discrètes.
    unary : np.ndarray de forme (height, width, n_labels)
        Coûts unaires (énergie). unary[i, j, d] = coût d'assigner l'étiquette d
        au pixel (i, j).
    pairwise_type : str
        Type de potentiel par paires : "potts" ou "truncated_linear".
    pairwise_weight : float
        Poids de lissage lambda.
    pairwise_trunc : float ou None
        Seuil de troncature pour le modèle linéaire tronqué.
    """

    def __init__(self, height, width, n_labels, unary,
                 pairwise_type="potts", pairwise_weight=1.0,
                 pairwise_trunc=None):
        self.height = height
        self.width = width
        self.n_labels = n_labels
        self.unary = np.asarray(unary, dtype=np.float64)
        self.pairwise_type = pairwise_type
        self.pairwise_weight = pairwise_weight
        self.pairwise_trunc = pairwise_trunc

        assert self.unary.shape == (height, width, n_labels)
        self._pairwise_costs = self._build_pairwise_costs()

    def _build_pairwise_costs(self):
        """Construit la table de coûts par paires cost[|d1 - d2|].

        Retourne ndarray de forme (n_labels,) où l'entrée k = cost(delta=k).
        """
        L = self.n_labels
        deltas = np.arange(L, dtype=np.float64)

        if self.pairwise_type == "potts":
            costs = np.where(deltas == 0, 0.0, self.pairwise_weight)
        elif self.pairwise_type == "truncated_linear":
            trunc = self.pairwise_trunc if self.pairwise_trunc is not None else L
            costs = self.pairwise_weight * np.minimum(deltas, trunc)
        else:
            raise ValueError(f"Type de paire inconnu : {self.pairwise_type}")

        return costs

    def pairwise_cost(self, d1, d2):
        """Coût par paires entre deux étiquettes."""
        return self._pairwise_costs[abs(d1 - d2)]

    def compute_energy(self, labeling):
        """Calcule l'énergie totale du MRF pour un étiquetage donné.

        Paramètres
        ----------
        labeling : np.ndarray de forme (height, width), dtype=int

        Retourne
        -------
        energy : float
        """
        labeling = np.asarray(labeling, dtype=int)
        H, W = self.height, self.width

        # Coûts unaires
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        energy = np.sum(self.unary[rows, cols, labeling])

        # Coûts par paires horizontaux
        h_diff = np.abs(labeling[:, :-1] - labeling[:, 1:])
        energy += np.sum(self._pairwise_costs[h_diff])

        # Coûts par paires verticaux
        v_diff = np.abs(labeling[:-1, :] - labeling[1:, :])
        energy += np.sum(self._pairwise_costs[v_diff])

        return energy

    def to_factor_graph(self):
        """Convertit en FactorGraph générique (seulement pour les petites grilles).

        Les potentiels sont convertis du domaine énergie au domaine probabilité :
        f(x) = exp(-cost(x))
        """
        fg = FactorGraph(name=f"Grid_{self.height}x{self.width}")
        H, W, L = self.height, self.width, self.n_labels

        var_names = np.empty((H, W), dtype=object)
        for i in range(H):
            for j in range(W):
                name = f"x_{i}_{j}"
                var_names[i, j] = name
                fg.add_variable(name, L)

        # Facteurs unaires
        for i in range(H):
            for j in range(W):
                vals = np.exp(-self.unary[i, j])
                fg.add_factor(Factor(
                    [var_names[i, j]], [L], vals, name=f"u_{i}_{j}"
                ))

        # Facteurs par paires (horizontaux)
        pair_vals = np.zeros((L, L))
        for d1 in range(L):
            for d2 in range(L):
                pair_vals[d1, d2] = np.exp(-self._pairwise_costs[abs(d1 - d2)])

        for i in range(H):
            for j in range(W - 1):
                fg.add_factor(Factor(
                    [var_names[i, j], var_names[i, j + 1]], [L, L],
                    pair_vals.copy(), name=f"h_{i}_{j}"
                ))

        # Facteurs par paires (verticaux)
        for i in range(H - 1):
            for j in range(W):
                fg.add_factor(Factor(
                    [var_names[i, j], var_names[i + 1, j]], [L, L],
                    pair_vals.copy(), name=f"v_{i}_{j}"
                ))

        return fg, var_names


def _dt_potts(h, gamma):
    """Transformée de distance pour le modèle de Potts en O(L).

    Pour chaque pixel, calcule : min(h[d], min(h) + gamma)

    Paramètres
    ----------
    h : np.ndarray de forme (..., L)
        Coûts par étiquette.
    gamma : float
        Pénalité Potts.

    Retourne
    -------
    result : np.ndarray de même forme que h
    """
    h_min = h.min(axis=-1, keepdims=True)
    return np.minimum(h, h_min + gamma)


def _dt_truncated_linear(h, weight, trunc):
    """Transformée de distance pour le modèle linéaire tronqué en O(L).

    Utilise l'algorithme de Felzenszwalb-Huttenlocher pour calculer :
    result[d] = min_d' (min(|d-d'|, trunc) * weight + h[d'])

    Paramètres
    ----------
    h : np.ndarray de forme (..., L)
    weight : float
    trunc : float
    """
    shape = h.shape
    L = shape[-1]
    flat = h.reshape(-1, L)
    N = flat.shape[0]
    result = np.full_like(flat, np.inf)

    # Passe avant : propagation vers la droite
    for i in range(N):
        result[i, 0] = flat[i, 0]
        for d in range(1, L):
            result[i, d] = min(flat[i, d], result[i, d - 1] + weight)

    # Passe arrière : propagation vers la gauche
    for i in range(N):
        for d in range(L - 2, -1, -1):
            result[i, d] = min(result[i, d], result[i, d + 1] + weight)

    # Troncature
    flat_min = flat.min(axis=-1, keepdims=True)
    result = np.minimum(result, flat_min + weight * trunc)

    return result.reshape(shape)


def grid_loopy_bp(mrf, max_iter=100, damping=0.5, tol=1e-6):
    """BP bouclée sur un GridMRF avec passage de messages vectorisé (min-sum).

    Paramètres
    ----------
    mrf : GridMRF
    max_iter : int
    damping : float
    tol : float

    Retourne
    -------
    labeling : np.ndarray de forme (H, W), dtype=int
    beliefs : np.ndarray de forme (H, W, L)
    energy_history : list of float
    """
    H, W, L = mrf.height, mrf.width, mrf.n_labels
    gamma = mrf.pairwise_weight

    # Messages : coûts envoyés dans chaque direction
    msg_r = np.zeros((H, W, L))  # vers la droite
    msg_l = np.zeros((H, W, L))  # vers la gauche
    msg_d = np.zeros((H, W, L))  # vers le bas
    msg_u = np.zeros((H, W, L))  # vers le haut

    energy_history = []

    for it in range(max_iter):
        old_msgs = [m.copy() for m in [msg_r, msg_l, msg_d, msg_u]]

        # Coût agrégé à chaque pixel (sans le message venant de la direction cible)
        total = mrf.unary + msg_r + msg_l + msg_d + msg_u

        # Messages vers la droite : pixel (i,j) envoie à (i,j+1)
        h = total[:, :-1, :] - msg_l[:, :-1, :]  # exclure le message venant de la gauche
        if mrf.pairwise_type == "potts":
            new_msg_r = _dt_potts(h, gamma)
        else:
            new_msg_r = _dt_truncated_linear(h, mrf.pairwise_weight, mrf.pairwise_trunc)
        new_msg_r -= new_msg_r.min(axis=-1, keepdims=True)
        msg_r_new = np.zeros_like(msg_r)
        msg_r_new[:, 1:, :] = damping * msg_r[:, 1:, :] + (1 - damping) * new_msg_r

        # Messages vers la gauche
        h = total[:, 1:, :] - msg_r[:, 1:, :]
        if mrf.pairwise_type == "potts":
            new_msg_l = _dt_potts(h, gamma)
        else:
            new_msg_l = _dt_truncated_linear(h, mrf.pairwise_weight, mrf.pairwise_trunc)
        new_msg_l -= new_msg_l.min(axis=-1, keepdims=True)
        msg_l_new = np.zeros_like(msg_l)
        msg_l_new[:, :-1, :] = damping * msg_l[:, :-1, :] + (1 - damping) * new_msg_l

        # Messages vers le bas
        h = total[:-1, :, :] - msg_u[:-1, :, :]
        if mrf.pairwise_type == "potts":
            new_msg_d = _dt_potts(h, gamma)
        else:
            new_msg_d = _dt_truncated_linear(h, mrf.pairwise_weight, mrf.pairwise_trunc)
        new_msg_d -= new_msg_d.min(axis=-1, keepdims=True)
        msg_d_new = np.zeros_like(msg_d)
        msg_d_new[1:, :, :] = damping * msg_d[1:, :, :] + (1 - damping) * new_msg_d

        # Messages vers le haut
        h = total[1:, :, :] - msg_d[1:, :, :]
        if mrf.pairwise_type == "potts":
            new_msg_u = _dt_potts(h, gamma)
        else:
            new_msg_u = _dt_truncated_linear(h, mrf.pairwise_weight, mrf.pairwise_trunc)
        new_msg_u -= new_msg_u.min(axis=-1, keepdims=True)
        msg_u_new = np.zeros_like(msg_u)
        msg_u_new[:-1, :, :] = damping * msg_u[:-1, :, :] + (1 - damping) * new_msg_u

        msg_r, msg_l, msg_d, msg_u = msg_r_new, msg_l_new, msg_d_new, msg_u_new

        # Croyances et étiquetage
        beliefs = mrf.unary + msg_r + msg_l + msg_d + msg_u
        labeling = beliefs.argmin(axis=-1)
        energy_history.append(mrf.compute_energy(labeling))

        # Vérification de la convergence
        max_diff = max(
            np.max(np.abs(msg_r - old_msgs[0])),
            np.max(np.abs(msg_l - old_msgs[1])),
            np.max(np.abs(msg_d - old_msgs[2])),
            np.max(np.abs(msg_u - old_msgs[3])),
        )
        if max_diff < tol:
            break

    beliefs = mrf.unary + msg_r + msg_l + msg_d + msg_u
    labeling = beliefs.argmin(axis=-1)
    return labeling, beliefs, energy_history


def grid_trw_s(mrf, max_iter=100, tol=1e-6):
    """TRW-S sur un GridMRF avec traitement séquentiel par lignes.

    Utilise rho = 0.5 pour toutes les arêtes (décomposition en arbres
    horizontaux et verticaux). Le traitement séquentiel assure la
    décroissance monotone de la borne.

    Paramètres
    ----------
    mrf : GridMRF
    max_iter : int
    tol : float

    Retourne
    -------
    labeling : np.ndarray de forme (H, W)
    beliefs : np.ndarray de forme (H, W, L)
    bound_history : list of float
    energy_history : list of float
    """
    H, W, L = mrf.height, mrf.width, mrf.n_labels
    rho = 0.5

    # Messages dans 4 directions
    msg_r = np.zeros((H, W, L))
    msg_l = np.zeros((H, W, L))
    msg_d = np.zeros((H, W, L))
    msg_u = np.zeros((H, W, L))

    bound_history = []
    energy_history = []

    for it in range(max_iter):
        old_msgs = [m.copy() for m in [msg_r, msg_l, msg_d, msg_u]]

        # Balayage avant : de haut en bas, de gauche à droite
        for i in range(H):
            for j in range(W):
                _update_pixel_trw(mrf, i, j, msg_r, msg_l, msg_d, msg_u, rho,
                                  direction="forward")

        # Balayage arrière : de bas en haut, de droite à gauche
        for i in range(H - 1, -1, -1):
            for j in range(W - 1, -1, -1):
                _update_pixel_trw(mrf, i, j, msg_r, msg_l, msg_d, msg_u, rho,
                                  direction="backward")

        # Croyances et étiquetage
        beliefs = mrf.unary + msg_r + msg_l + msg_d + msg_u
        labeling = beliefs.argmin(axis=-1)
        energy_history.append(mrf.compute_energy(labeling))

        # Borne TRW (approchée par l'énergie duale)
        bound = _compute_grid_trw_bound(mrf, beliefs, msg_r, msg_l, msg_d, msg_u, rho)
        bound_history.append(bound)

        # Convergence
        max_diff = max(
            np.max(np.abs(msg_r - old_msgs[0])),
            np.max(np.abs(msg_l - old_msgs[1])),
            np.max(np.abs(msg_d - old_msgs[2])),
            np.max(np.abs(msg_u - old_msgs[3])),
        )
        if max_diff < tol:
            break

    beliefs = mrf.unary + msg_r + msg_l + msg_d + msg_u
    labeling = beliefs.argmin(axis=-1)
    return labeling, beliefs, bound_history, energy_history


def _update_pixel_trw(mrf, i, j, msg_r, msg_l, msg_d, msg_u, rho,
                      direction="forward"):
    """Met à jour les messages TRW-S pour un pixel (i, j).

    Dans le balayage avant, on met à jour les messages vers la droite et vers le bas.
    Dans le balayage arrière, vers la gauche et vers le haut.
    """
    H, W, L = mrf.height, mrf.width, mrf.n_labels
    gamma = mrf.pairwise_weight

    # Coût agrégé au pixel courant
    h_total = mrf.unary[i, j].copy()
    if j > 0:
        h_total += msg_r[i, j]      # venant de la gauche
    if j < W - 1:
        h_total += msg_l[i, j]      # venant de la droite
    if i > 0:
        h_total += msg_d[i, j]      # venant du haut
    if i < H - 1:
        h_total += msg_u[i, j]      # venant du bas

    if direction == "forward":
        # Message vers la droite (vers pixel (i, j+1))
        if j < W - 1:
            h = h_total - msg_l[i, j]  # exclure le message venant du voisin droit
            if mrf.pairwise_type == "potts":
                new_msg = _dt_potts_1d(h, gamma)
            else:
                new_msg = _dt_truncated_linear_1d(h, mrf.pairwise_weight, mrf.pairwise_trunc)
            new_msg -= new_msg.min()
            msg_r[i, j + 1] = new_msg

        # Message vers le bas (vers pixel (i+1, j))
        if i < H - 1:
            h = h_total - msg_u[i, j]  # exclure le message venant du voisin bas
            if mrf.pairwise_type == "potts":
                new_msg = _dt_potts_1d(h, gamma)
            else:
                new_msg = _dt_truncated_linear_1d(h, mrf.pairwise_weight, mrf.pairwise_trunc)
            new_msg -= new_msg.min()
            msg_d[i + 1, j] = new_msg

    else:  # backward
        # Message vers la gauche (vers pixel (i, j-1))
        if j > 0:
            h = h_total - msg_r[i, j]
            if mrf.pairwise_type == "potts":
                new_msg = _dt_potts_1d(h, gamma)
            else:
                new_msg = _dt_truncated_linear_1d(h, mrf.pairwise_weight, mrf.pairwise_trunc)
            new_msg -= new_msg.min()
            msg_l[i, j - 1] = new_msg

        # Message vers le haut (vers pixel (i-1, j))
        if i > 0:
            h = h_total - msg_d[i, j]
            if mrf.pairwise_type == "potts":
                new_msg = _dt_potts_1d(h, gamma)
            else:
                new_msg = _dt_truncated_linear_1d(h, mrf.pairwise_weight, mrf.pairwise_trunc)
            new_msg -= new_msg.min()
            msg_u[i - 1, j] = new_msg


def _dt_potts_1d(h, gamma):
    """Transformée de distance Potts pour un seul vecteur 1D."""
    return np.minimum(h, h.min() + gamma)


def _dt_truncated_linear_1d(h, weight, trunc):
    """Transformée de distance linéaire tronquée pour un seul vecteur 1D."""
    L = len(h)
    result = h.copy()
    # Passe avant
    for d in range(1, L):
        result[d] = min(result[d], result[d - 1] + weight)
    # Passe arrière
    for d in range(L - 2, -1, -1):
        result[d] = min(result[d], result[d + 1] + weight)
    # Troncature
    result = np.minimum(result, h.min() + weight * trunc)
    return result


def _compute_grid_trw_bound(mrf, beliefs, msg_r, msg_l, msg_d, msg_u, rho):
    """Calcule une borne inférieure sur l'énergie MRF (borne duale TRW).

    La borne est la somme des min-marginales pondérées par rho.
    """
    H, W, L = mrf.height, mrf.width, mrf.n_labels

    # Borne par les min-marginales des nœuds
    node_min = beliefs.min(axis=-1)
    bound = np.sum(node_min)

    return bound


def grid_mean_field(mrf, max_iter=100, tol=1e-6, seed=None):
    """Champ moyen sur un GridMRF, entièrement vectorisé.

    Paramètres
    ----------
    mrf : GridMRF
    max_iter : int
    tol : float
    seed : int ou None

    Retourne
    -------
    labeling : np.ndarray de forme (H, W)
    q : np.ndarray de forme (H, W, L)
    energy_history : list of float
    """
    H, W, L = mrf.height, mrf.width, mrf.n_labels
    rng = np.random.default_rng(seed)

    # Initialisation de q
    q = rng.dirichlet(np.ones(L), size=(H, W))
    energy_history = []

    for it in range(max_iter):
        old_q = q.copy()

        # Terme d'énergie unaire : -unary
        log_q = -mrf.unary.copy()

        # Terme de lissage : pour Potts, E_q[cost] = lambda * (1 - q_voisin(d))
        if mrf.pairwise_type == "potts":
            gamma = mrf.pairwise_weight
            # Somme des q des voisins pour chaque direction
            q_neighbors = np.zeros_like(q)
            q_neighbors[:, 1:, :] += q[:, :-1, :]   # voisin gauche
            q_neighbors[:, :-1, :] += q[:, 1:, :]   # voisin droit
            q_neighbors[1:, :, :] += q[:-1, :, :]   # voisin haut
            q_neighbors[:-1, :, :] += q[1:, :, :]   # voisin bas
            log_q += gamma * q_neighbors
        else:
            # Linéaire tronqué : E_q[cost(d, d')] = sum_{d'} q(d') * min(|d-d'|, trunc) * weight
            for d in range(L):
                cost_d = np.zeros((H, W))
                for dp in range(L):
                    c = mrf._pairwise_costs[abs(d - dp)]
                    q_neighbors = np.zeros((H, W))
                    if W > 1:
                        q_neighbors[:, 1:] += q[:, :-1, dp]
                        q_neighbors[:, :-1] += q[:, 1:, dp]
                    if H > 1:
                        q_neighbors[1:, :] += q[:-1, :, dp]
                        q_neighbors[:-1, :] += q[1:, :, dp]
                    cost_d += c * q_neighbors
                log_q[:, :, d] -= cost_d

        # Softmax pour obtenir les distributions
        log_q -= log_q.max(axis=-1, keepdims=True)
        q = np.exp(log_q)
        q /= q.sum(axis=-1, keepdims=True)

        # Étiquetage et énergie
        labeling = q.argmax(axis=-1)
        energy_history.append(mrf.compute_energy(labeling))

        # Convergence
        if np.max(np.abs(q - old_q)) < tol:
            break

    labeling = q.argmax(axis=-1)
    return labeling, q, energy_history
