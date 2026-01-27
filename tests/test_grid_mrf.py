"""Tests pour le GridMRF et ses algorithmes d'inférence."""

import numpy as np
import pytest
from src.grid_mrf import GridMRF, grid_loopy_bp, grid_trw_s, grid_mean_field
from src.belief_propagation import loopy_bp
from src.variational import mean_field_vi
from src.trw import trw_s


@pytest.fixture
def small_grid_mrf():
    """GridMRF 3x3 avec 3 étiquettes pour les tests de validation."""
    rng = np.random.default_rng(42)
    H, W, L = 3, 3, 3
    unary = rng.uniform(0, 5, size=(H, W, L))
    return GridMRF(H, W, L, unary, pairwise_type="potts", pairwise_weight=1.0)


@pytest.fixture
def tiny_grid_mrf():
    """GridMRF 2x2 binaire minuscule pour des tests rapides."""
    unary = np.array([
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.5, 0.5], [0.5, 1.5]],
    ])
    return GridMRF(2, 2, 2, unary, pairwise_type="potts", pairwise_weight=0.5)


class TestGridMRF:
    """Tests pour la classe GridMRF."""

    def test_energy_consistent(self, small_grid_mrf):
        """L'énergie doit être calculable pour tout étiquetage."""
        labeling = np.zeros((3, 3), dtype=int)
        energy = small_grid_mrf.compute_energy(labeling)
        assert np.isfinite(energy)

    def test_uniform_labeling_lower_energy_with_potts(self, small_grid_mrf):
        """Un étiquetage uniforme doit avoir un coût de lissage nul pour Potts."""
        uniform = np.zeros((3, 3), dtype=int)
        energy_uniform = small_grid_mrf.compute_energy(uniform)

        # Étiquetage alterné (coût de lissage élevé)
        alternating = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        energy_alt = small_grid_mrf.compute_energy(alternating)

        # Le coût de lissage est 0 pour uniform, > 0 pour alternating
        # Mais le coût unaire peut compenser, donc on vérifie juste que la différence est logique
        pairwise_cost_alt = energy_alt - np.sum(
            small_grid_mrf.unary[np.arange(3)[:, None], np.arange(3), alternating]
        )
        assert pairwise_cost_alt > 0

    def test_to_factor_graph_roundtrip(self, tiny_grid_mrf):
        """La conversion vers FactorGraph doit être cohérente."""
        fg, var_names = tiny_grid_mrf.to_factor_graph()
        assert fg.n_variables == 4  # 2x2
        # Unaires + pairwise : 4 unaires + 2 horiz + 2 vert = 8
        assert fg.n_factors == 8

    def test_pairwise_costs_potts(self):
        """Le coût Potts doit être 0 si d1==d2, lambda sinon."""
        mrf = GridMRF(2, 2, 3, np.zeros((2, 2, 3)),
                      pairwise_type="potts", pairwise_weight=2.0)
        assert mrf.pairwise_cost(0, 0) == 0.0
        assert mrf.pairwise_cost(0, 1) == 2.0
        assert mrf.pairwise_cost(0, 2) == 2.0

    def test_pairwise_costs_truncated_linear(self):
        """Le coût linéaire tronqué doit être min(|d1-d2|, trunc) * weight."""
        mrf = GridMRF(2, 2, 5, np.zeros((2, 2, 5)),
                      pairwise_type="truncated_linear",
                      pairwise_weight=1.5, pairwise_trunc=2.0)
        assert mrf.pairwise_cost(0, 0) == 0.0
        assert mrf.pairwise_cost(0, 1) == 1.5
        assert mrf.pairwise_cost(0, 2) == 3.0
        assert mrf.pairwise_cost(0, 3) == 3.0  # tronqué
        assert mrf.pairwise_cost(0, 4) == 3.0  # tronqué


class TestGridLoopyBP:
    """Tests pour la BP bouclée sur grille."""

    def test_returns_valid_labeling(self, small_grid_mrf):
        """Le résultat doit être un étiquetage valide."""
        labeling, beliefs, energy_hist = grid_loopy_bp(
            small_grid_mrf, max_iter=20, damping=0.5
        )
        assert labeling.shape == (3, 3)
        assert np.all(labeling >= 0) and np.all(labeling < 3)
        assert len(energy_hist) > 0

    def test_energy_decreases_or_stable(self, small_grid_mrf):
        """L'énergie devrait globalement diminuer ou se stabiliser."""
        _, _, energy_hist = grid_loopy_bp(
            small_grid_mrf, max_iter=50, damping=0.5
        )
        # Vérifier que l'énergie finale est <= énergie initiale (avec tolérance)
        assert energy_hist[-1] <= energy_hist[0] + 1.0

    def test_strong_unary_respected(self):
        """Avec des coûts unaires très forts, l'étiquetage doit suivre les unaires."""
        H, W, L = 3, 3, 3
        unary = np.full((H, W, L), 100.0)
        # Mettre le coût de l'étiquette 1 à 0 partout
        unary[:, :, 1] = 0.0
        mrf = GridMRF(H, W, L, unary, pairwise_type="potts", pairwise_weight=1.0)
        labeling, _, _ = grid_loopy_bp(mrf, max_iter=20, damping=0.5)
        np.testing.assert_array_equal(labeling, 1)


class TestGridTRWS:
    """Tests pour TRW-S sur grille."""

    def test_returns_valid_labeling(self, small_grid_mrf):
        """Le résultat doit être un étiquetage valide."""
        labeling, beliefs, bound_hist, energy_hist = grid_trw_s(
            small_grid_mrf, max_iter=20
        )
        assert labeling.shape == (3, 3)
        assert np.all(labeling >= 0) and np.all(labeling < 3)
        assert len(bound_hist) > 0
        assert len(energy_hist) > 0

    def test_bound_history_nonempty(self, small_grid_mrf):
        """L'historique de la borne ne doit pas être vide."""
        _, _, bound_hist, _ = grid_trw_s(small_grid_mrf, max_iter=10)
        assert len(bound_hist) > 0
        assert all(np.isfinite(b) for b in bound_hist)

    def test_strong_unary_respected(self):
        """Avec des coûts unaires très forts, l'étiquetage doit suivre les unaires."""
        H, W, L = 3, 3, 3
        unary = np.full((H, W, L), 100.0)
        unary[:, :, 2] = 0.0
        mrf = GridMRF(H, W, L, unary, pairwise_type="potts", pairwise_weight=1.0)
        labeling, _, _, _ = grid_trw_s(mrf, max_iter=20)
        np.testing.assert_array_equal(labeling, 2)


class TestGridMeanField:
    """Tests pour le champ moyen sur grille."""

    def test_returns_valid_labeling(self, small_grid_mrf):
        """Le résultat doit être un étiquetage valide."""
        labeling, q, energy_hist = grid_mean_field(
            small_grid_mrf, max_iter=20, seed=42
        )
        assert labeling.shape == (3, 3)
        assert np.all(labeling >= 0) and np.all(labeling < 3)
        assert q.shape == (3, 3, 3)

    def test_q_is_valid_distribution(self, small_grid_mrf):
        """q doit être des distributions valides (non-négatif, somme à 1)."""
        _, q, _ = grid_mean_field(small_grid_mrf, max_iter=20, seed=42)
        assert np.all(q >= 0)
        np.testing.assert_allclose(q.sum(axis=-1), 1.0, atol=1e-8)

    def test_strong_unary_respected(self):
        """Avec des coûts unaires très forts, l'étiquetage doit suivre les unaires."""
        H, W, L = 3, 3, 3
        unary = np.full((H, W, L), 100.0)
        unary[:, :, 0] = 0.0
        mrf = GridMRF(H, W, L, unary, pairwise_type="potts", pairwise_weight=1.0)
        labeling, _, _ = grid_mean_field(mrf, max_iter=20, seed=42)
        np.testing.assert_array_equal(labeling, 0)


class TestCrossValidation:
    """Validation croisée entre GridMRF et FactorGraph générique."""

    def test_energy_matches(self, tiny_grid_mrf):
        """L'énergie doit correspondre entre GridMRF et FactorGraph."""
        from src.trw import compute_mrf_energy

        fg, var_names = tiny_grid_mrf.to_factor_graph()
        labeling = np.array([[0, 1], [1, 0]])

        grid_energy = tiny_grid_mrf.compute_energy(labeling)

        assignment = {}
        for i in range(2):
            for j in range(2):
                assignment[var_names[i, j]] = labeling[i, j]
        fg_energy = compute_mrf_energy(fg, assignment)

        np.testing.assert_allclose(grid_energy, fg_energy, atol=1e-8)
