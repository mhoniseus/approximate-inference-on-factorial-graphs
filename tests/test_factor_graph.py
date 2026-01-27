"""Tests pour les structures de données de graphes factoriels."""

import numpy as np
import pytest
from src.factor_graph import Factor, FactorGraph


class TestFactor:
    """Tests pour la classe Factor."""

    def test_default_values_are_ones(self):
        """Un facteur sans valeurs doit être initialisé à des uns."""
        f = Factor(["a", "b"], [2, 3])
        assert f.values.shape == (2, 3)
        assert np.allclose(f.values, 1.0)

    def test_custom_values_shape(self):
        """Les valeurs personnalisées doivent correspondre aux cardinalités déclarées."""
        vals = np.array([[1.0, 2.0], [3.0, 4.0]])
        f = Factor(["x", "y"], [2, 2], vals)
        assert f.values.shape == (2, 2)
        assert np.isclose(f.values[0, 1], 2.0)

    def test_wrong_shape_raises(self):
        """Fournir des valeurs avec la mauvaise forme doit lever une erreur."""
        with pytest.raises(AssertionError):
            Factor(["x", "y"], [2, 3], np.ones((2, 2)))

    def test_scope(self):
        """La portée du facteur doit être l'ensemble de ses noms de variables."""
        f = Factor(["a", "b", "c"], [2, 2, 2])
        assert f.scope == {"a", "b", "c"}

    def test_marginalize_reduces_dimension(self):
        """Marginaliser une variable doit la retirer de la portée."""
        vals = np.array([[1.0, 2.0], [3.0, 4.0]])
        f = Factor(["x", "y"], [2, 2], vals)
        fm = f.marginalize("x")
        assert fm.variables == ["y"]
        assert fm.values.shape == (2,)
        # Somme sur x : [1+3, 2+4] = [4, 6]
        assert np.allclose(fm.values, [4.0, 6.0])

    def test_maximize_reduces_dimension(self):
        """Maximiser une variable doit la retirer et prendre le max."""
        vals = np.array([[1.0, 5.0], [3.0, 2.0]])
        f = Factor(["x", "y"], [2, 2], vals)
        fm = f.maximize("x")
        assert fm.variables == ["y"]
        # Max sur x : [max(1,3), max(5,2)] = [3, 5]
        assert np.allclose(fm.values, [3.0, 5.0])

    def test_multiply_shared_variable(self):
        """Multiplication de deux facteurs avec une variable partagée."""
        f1 = Factor(["x", "y"], [2, 2], np.array([[1.0, 2.0], [3.0, 4.0]]))
        f2 = Factor(["y", "z"], [2, 2], np.array([[5.0, 6.0], [7.0, 8.0]]))
        prod = f1.multiply(f2)
        assert set(prod.variables) == {"x", "y", "z"}
        assert prod.values.shape == (2, 2, 2)

    def test_multiply_disjoint_variables(self):
        """Multiplication de facteurs sans variables partagées (produit extérieur)."""
        f1 = Factor(["a"], [2], np.array([2.0, 3.0]))
        f2 = Factor(["b"], [3], np.array([1.0, 4.0, 5.0]))
        prod = f1.multiply(f2)
        assert set(prod.variables) == {"a", "b"}
        assert prod.values.shape == (2, 3)
        assert np.isclose(prod.values[1, 2], 15.0)  # 3 * 5

    def test_normalize(self):
        """La normalisation doit faire en sorte que la table somme à 1."""
        f = Factor(["x"], [4], np.array([1.0, 2.0, 3.0, 4.0]))
        f.normalize()
        assert np.isclose(f.values.sum(), 1.0)


class TestFactorGraph:
    """Tests pour la classe FactorGraph."""

    def test_add_variable(self):
        """Les variables doivent être enregistrées avec leurs cardinalités."""
        fg = FactorGraph()
        fg.add_variable("x", 3)
        assert fg.n_variables == 1
        assert fg.variables["x"] == 3

    def test_add_factor_connects_variables(self):
        """L'ajout d'un facteur doit le lier à ses variables."""
        fg = FactorGraph()
        fg.add_variable("x", 2)
        fg.add_variable("y", 2)
        f = Factor(["x", "y"], [2, 2])
        fg.add_factor(f)
        assert fg.n_factors == 1
        assert 0 in fg.neighbors_of_variable("x")
        assert 0 in fg.neighbors_of_variable("y")

    def test_unregistered_variable_raises(self):
        """L'ajout d'un facteur avec une variable inconnue doit lever une erreur."""
        fg = FactorGraph()
        fg.add_variable("x", 2)
        f = Factor(["x", "z"], [2, 2])
        with pytest.raises(ValueError, match="non enregistrée"):
            fg.add_factor(f)

    def test_neighbors_of_factor(self):
        """Les voisins d'un facteur sont ses noms de variables."""
        fg = FactorGraph()
        fg.add_variable("a", 2)
        fg.add_variable("b", 3)
        f = Factor(["a", "b"], [2, 3])
        fg.add_factor(f)
        assert set(fg.neighbors_of_factor(0)) == {"a", "b"}

    def test_chain_is_tree(self):
        """Un graphe factoriel en chaîne doit être détecté comme un arbre."""
        fg = FactorGraph()
        for i in range(4):
            fg.add_variable(f"x{i}", 2)
        for i in range(3):
            fg.add_factor(Factor([f"x{i}", f"x{i+1}"], [2, 2]))
        assert fg.is_tree()

    def test_grid_is_not_tree(self):
        """Une grille 2x2 a des cycles et ne doit pas être un arbre."""
        fg = FactorGraph()
        for name in ["a", "b", "c", "d"]:
            fg.add_variable(name, 2)
        # a-b, b-d, a-c, c-d forme un cycle
        for pair in [("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")]:
            fg.add_factor(Factor(list(pair), [2, 2]))
        assert not fg.is_tree()

    def test_to_networkx(self):
        """La conversion en NetworkX doit produire le bon nombre de nœuds."""
        fg = FactorGraph()
        fg.add_variable("x", 2)
        fg.add_variable("y", 2)
        f = Factor(["x", "y"], [2, 2], name="f_xy")
        fg.add_factor(f)
        G = fg.to_networkx()
        # 2 nœuds variables + 1 nœud facteur
        assert len(G.nodes) == 3
        assert len(G.edges) == 2

    def test_joint_distribution_sums_to_one(self, binary_chain):
        """La distribution jointe d'un petit graphe doit sommer à 1."""
        joint = binary_chain.joint_distribution()
        assert np.isclose(joint.values.sum(), 1.0)
