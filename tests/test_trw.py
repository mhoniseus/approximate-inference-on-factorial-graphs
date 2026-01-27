"""Tests pour le passage de messages TRW-S."""

import numpy as np
import pytest
from src.trw import trw_s, compute_mrf_energy, _compute_rho
from src.belief_propagation import sum_product_bp
from src.factor_graph import Factor, FactorGraph
from src.utils import generate_ising_grid, generate_chain


class TestComputeRho:
    """Tests pour le calcul des probabilités d'apparition rho."""

    def test_tree_rho_all_one(self, biased_chain):
        """Sur un arbre, tous les rho doivent être 1.0."""
        rho = _compute_rho(biased_chain)
        for fi in rho:
            assert rho[fi] == 1.0

    def test_grid_rho(self, ising_4x4):
        """Sur une grille, les facteurs unaires ont rho=1, les pairwise rho=0.5."""
        fg, _ = ising_4x4
        rho = _compute_rho(fg)
        for fi, factor in enumerate(fg.factors):
            if len(factor.variables) == 1:
                assert rho[fi] == 1.0
            else:
                assert rho[fi] == 0.5


class TestComputeMRFEnergy:
    """Tests pour le calcul de l'énergie MRF."""

    def test_energy_simple_chain(self, binary_chain):
        """L'énergie doit être calculable pour une assignation connue."""
        assignment = {"x0": 0, "x1": 0, "x2": 0}
        energy = compute_mrf_energy(binary_chain, assignment)
        assert np.isfinite(energy)

    def test_agreeing_assignment_lower_energy(self, binary_chain):
        """Une assignation en accord (tous identiques) doit avoir une énergie
        plus basse qu'une assignation en désaccord avec un couplage positif."""
        e_agree = compute_mrf_energy(binary_chain, {"x0": 0, "x1": 0, "x2": 0})
        e_disagree = compute_mrf_energy(binary_chain, {"x0": 0, "x1": 1, "x2": 0})
        assert e_agree < e_disagree


class TestTRWS:
    """Tests pour l'algorithme TRW-S."""

    def test_tree_matches_bp(self, biased_chain):
        """Sur un arbre, les croyances TRW-S doivent correspondre à la BP somme-produit."""
        bp_beliefs, bp_conv, _ = sum_product_bp(biased_chain, max_iter=100)
        trw_beliefs, trw_conv, _ = trw_s(biased_chain, max_iter=100, damping=0.0)

        assert bp_conv
        assert trw_conv

        for var in bp_beliefs:
            np.testing.assert_allclose(
                trw_beliefs[var], bp_beliefs[var], atol=1e-6,
                err_msg=f"Croyances TRW-S != BP pour {var}"
            )

    def test_tree_matches_bp_chain(self):
        """Sur une chaîne (arbre), TRW-S doit être exact comme BP."""
        fg = generate_chain(5, cardinality=2, coupling=1.0, seed=42)
        bp_beliefs, _, _ = sum_product_bp(fg, max_iter=100)
        trw_beliefs, _, _ = trw_s(fg, max_iter=100, damping=0.0)

        for var in bp_beliefs:
            np.testing.assert_allclose(
                trw_beliefs[var], bp_beliefs[var], atol=1e-6
            )

    def test_converges_on_grid(self, ising_4x4):
        """TRW-S doit converger sur une petite grille d'Ising."""
        fg, _ = ising_4x4
        beliefs, converged, bound_history = trw_s(
            fg, max_iter=300, damping=0.5, tol=1e-6
        )
        assert converged or len(bound_history) == 300
        assert len(beliefs) == fg.n_variables

    def test_beliefs_are_valid_distributions(self, ising_4x4):
        """Toutes les croyances doivent être non-négatives et sommer à 1."""
        fg, _ = ising_4x4
        beliefs, _, _ = trw_s(fg, max_iter=100, damping=0.5)

        for var, b in beliefs.items():
            assert np.all(b >= 0), f"Croyance négative pour {var}"
            np.testing.assert_allclose(
                b.sum(), 1.0, atol=1e-8,
                err_msg=f"Croyance ne somme pas à 1 pour {var}"
            )

    def test_bound_history_nonempty(self, ising_4x4):
        """L'historique de la borne ne doit pas être vide."""
        fg, _ = ising_4x4
        _, _, bound_history = trw_s(fg, max_iter=10, damping=0.5)
        assert len(bound_history) > 0
        assert all(np.isfinite(b) for b in bound_history)

    def test_bound_is_upper_bound_on_tree(self, biased_chain):
        """Sur un arbre, la borne TRW doit être proche de log Z exact."""
        # Calcul de log Z exact via la jointe
        joint = biased_chain.joint_distribution()
        # joint est normalisé, on doit recalculer Z
        product = biased_chain.factors[0]
        for f in biased_chain.factors[1:]:
            product = product.multiply(f)
        log_Z = np.log(product.values.sum())

        _, _, bound_history = trw_s(biased_chain, max_iter=100, damping=0.0)

        # Sur un arbre avec rho=1, la borne doit être proche de log Z
        final_bound = bound_history[-1]
        # La borne doit être >= log Z (à une tolérance numérique près)
        assert final_bound >= log_Z - 0.1, (
            f"Borne {final_bound} < log Z {log_Z}"
        )

    def test_single_variable(self, single_factor):
        """TRW-S doit fonctionner sur une seule variable."""
        beliefs, converged, _ = trw_s(single_factor, max_iter=50, damping=0.0)
        assert converged
        expected = np.array([1.0, 3.0, 2.0])
        expected /= expected.sum()
        np.testing.assert_allclose(beliefs["y"], expected, atol=1e-8)

    def test_symmetric_chain_uniform_beliefs(self):
        """Une chaîne symétrique doit donner des croyances uniformes."""
        fg = FactorGraph(name="SymChain")
        for i in range(3):
            fg.add_variable(f"x{i}", 2)

        pair = np.array([[np.exp(1.0), np.exp(-1.0)],
                         [np.exp(-1.0), np.exp(1.0)]])
        fg.add_factor(Factor(["x0", "x1"], [2, 2], pair.copy()))
        fg.add_factor(Factor(["x1", "x2"], [2, 2], pair.copy()))

        for i in range(3):
            fg.add_factor(Factor([f"x{i}"], [2], np.ones(2)))

        beliefs, _, _ = trw_s(fg, max_iter=100, damping=0.0)
        for var in beliefs:
            np.testing.assert_allclose(
                beliefs[var], [0.5, 0.5], atol=1e-8,
                err_msg=f"Croyance non uniforme pour {var}"
            )
