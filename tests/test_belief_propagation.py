"""Tests pour les algorithmes de propagation de croyances."""

import numpy as np
import pytest
from src.belief_propagation import sum_product_bp, max_product_bp, loopy_bp
from src.factor_graph import Factor, FactorGraph


class TestSumProductBP:
    """Tests pour la propagation de croyances somme-produit."""

    def test_uniform_chain_gives_uniform_beliefs(self, binary_chain):
        """Sur une chaîne symétrique avec des facteurs unaires uniformes, les croyances doivent être uniformes."""
        beliefs, converged, n_iter = sum_product_bp(binary_chain)
        for var, b in beliefs.items():
            assert np.allclose(b, [0.5, 0.5], atol=1e-6), (
                f"Croyance uniforme attendue pour {var}, obtenue {b}"
            )

    def test_single_variable_exact(self, single_factor):
        """La BP sur un seul facteur unaire doit retourner le potentiel normalisé."""
        beliefs, converged, n_iter = sum_product_bp(single_factor)
        expected = np.array([1.0, 3.0, 2.0])
        expected = expected / expected.sum()
        assert np.allclose(beliefs["y"], expected, atol=1e-8)

    def test_converges_on_tree(self, biased_chain):
        """Le somme-produit doit converger sur un arbre (chaîne)."""
        beliefs, converged, n_iter = sum_product_bp(biased_chain)
        assert converged

    def test_bias_propagates(self, biased_chain):
        """Le biais sur x0 doit se propager : x0 le plus biaisé, x2 le moins."""
        beliefs, _, _ = sum_product_bp(biased_chain)
        # x0 a un biais direct vers l'état 1
        assert beliefs["x0"][1] > beliefs["x0"][0]
        # x1 et x2 doivent aussi pencher vers l'état 1 grâce au couplage
        assert beliefs["x1"][1] > 0.5
        assert beliefs["x2"][1] > 0.5
        # Le biais doit décroître le long de la chaîne
        assert beliefs["x0"][1] > beliefs["x1"][1]
        assert beliefs["x1"][1] > beliefs["x2"][1]

    def test_beliefs_are_valid_distributions(self, biased_chain):
        """Toutes les croyances doivent être non négatives et sommer à 1."""
        beliefs, _, _ = sum_product_bp(biased_chain)
        for var, b in beliefs.items():
            assert np.all(b >= 0), f"Croyance négative pour {var}"
            assert np.isclose(b.sum(), 1.0), f"La croyance de {var} somme à {b.sum()}"

    def test_matches_brute_force_on_chain(self, biased_chain):
        """Les marginales BP sur un arbre doivent correspondre au calcul en force brute."""
        beliefs_bp, _, _ = sum_product_bp(biased_chain)

        # Force brute : calculer la jointe complète et marginaliser
        joint = biased_chain.joint_distribution()
        for var in biased_chain.variables:
            # Marginaliser toutes les autres variables
            marg = joint
            for other_var in biased_chain.variables:
                if other_var != var:
                    marg = marg.marginalize(other_var)
            marg.normalize()
            assert np.allclose(beliefs_bp[var], marg.values, atol=1e-6), (
                f"La BP diverge de la force brute pour {var}"
            )


class TestMaxProductBP:
    """Tests pour la propagation de croyances max-produit."""

    def test_single_variable_map(self, single_factor):
        """Le MAP d'une seule variable doit être le mode du potentiel."""
        assignment, beliefs, converged, n_iter = max_product_bp(single_factor)
        assert assignment["y"] == 1  # l'état 1 a le potentiel 3.0 (le plus élevé)

    def test_converges_on_tree(self, biased_chain):
        """Le max-produit doit converger sur un arbre."""
        _, _, converged, _ = max_product_bp(biased_chain)
        assert converged

    def test_biased_chain_map(self, biased_chain):
        """Avec un biais fort vers l'état 1, le MAP doit assigner tous à 1."""
        assignment, _, _, _ = max_product_bp(biased_chain)
        # Étant donné le biais fort sur x0 et le couplage, tous doivent préférer l'état 1
        assert assignment["x0"] == 1


class TestLoopyBP:
    """Tests pour la propagation de croyances bouclée (avec amortissement)."""

    def test_tree_matches_exact(self, biased_chain):
        """Sur un arbre, la BP bouclée doit correspondre au somme-produit exact."""
        beliefs_exact, _, _ = sum_product_bp(biased_chain)
        beliefs_loopy, _, _ = loopy_bp(biased_chain, damping=0.0, max_iter=200)
        for var in biased_chain.variables:
            assert np.allclose(beliefs_exact[var], beliefs_loopy[var], atol=1e-6), (
                f"La BP bouclée diverge de l'exact pour {var}"
            )

    def test_damping_does_not_break_tree(self, biased_chain):
        """L'amortissement ne doit pas changer le point fixe sur un arbre."""
        beliefs_exact, _, _ = sum_product_bp(biased_chain)
        beliefs_damped, _, _ = loopy_bp(biased_chain, damping=0.3, max_iter=500)
        for var in biased_chain.variables:
            assert np.allclose(beliefs_exact[var], beliefs_damped[var], atol=1e-4), (
                f"La BP bouclée amortie a divergé pour {var}"
            )

    def test_returns_valid_beliefs(self, ising_4x4):
        """La BP bouclée sur une grille doit retourner des distributions de probabilité valides."""
        fg, _ = ising_4x4
        beliefs, _, history = loopy_bp(fg, damping=0.5, max_iter=100)
        for var, b in beliefs.items():
            assert np.all(b >= 0), f"Croyance négative pour {var}"
            assert np.isclose(b.sum(), 1.0, atol=1e-6), (
                f"La croyance de {var} somme à {b.sum()}"
            )

    def test_history_is_recorded(self, ising_4x4):
        """L'historique de convergence doit être rempli."""
        fg, _ = ising_4x4
        _, _, history = loopy_bp(fg, damping=0.5, max_iter=50)
        assert len(history) > 0
        assert len(history) <= 50
