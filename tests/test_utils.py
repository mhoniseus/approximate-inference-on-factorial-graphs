"""Tests pour les fonctions utilitaires."""

import numpy as np
import pytest
from src.utils import generate_ising_grid, generate_random_fg, generate_chain


class TestGenerateIsingGrid:
    """Tests pour la génération de grilles d'Ising."""

    def test_variable_count(self):
        """Une grille 3x4 doit avoir 12 variables."""
        fg, grid = generate_ising_grid(3, 4)
        assert fg.n_variables == 12
        assert grid.shape == (3, 4)

    def test_factor_count(self):
        """Une grille 3x4 doit avoir 2*3*4 - 3 - 4 = 17 paires + 12 unaires = 29 facteurs."""
        fg, _ = generate_ising_grid(3, 4)
        n_horizontal = 3 * 3  # 3 lignes, 3 arêtes horizontales chacune
        n_vertical = 2 * 4    # 2 écarts verticaux, 4 colonnes chacun
        n_pairwise = n_horizontal + n_vertical  # 9 + 8 = 17
        n_unary = 12
        assert fg.n_factors == n_pairwise + n_unary

    def test_all_binary(self):
        """Toutes les variables du modèle d'Ising doivent être binaires."""
        fg, _ = generate_ising_grid(5, 5)
        for var, card in fg.variables.items():
            assert card == 2, f"{var} a la cardinalité {card}"

    def test_deterministic_with_seed(self):
        """La même graine doit produire le même graphe factoriel."""
        fg1, _ = generate_ising_grid(3, 3, seed=42)
        fg2, _ = generate_ising_grid(3, 3, seed=42)
        for f1, f2 in zip(fg1.factors, fg2.factors):
            assert np.array_equal(f1.values, f2.values)


class TestGenerateRandomFG:
    """Tests pour la génération de graphes factoriels aléatoires."""

    def test_variable_count(self):
        """Doit créer le nombre demandé de variables."""
        fg = generate_random_fg(5, 3, seed=0)
        assert fg.n_variables == 5

    def test_factor_count(self):
        """Doit créer le nombre demandé de facteurs."""
        fg = generate_random_fg(5, 7, seed=0)
        assert fg.n_factors == 7

    def test_cardinalities_in_range(self):
        """Les cardinalités des variables doivent être dans [2, max_cardinality]."""
        fg = generate_random_fg(10, 5, max_cardinality=4, seed=0)
        for var, card in fg.variables.items():
            assert 2 <= card <= 4, f"{var} a la cardinalité {card}"


class TestGenerateChain:
    """Tests pour la génération de graphes factoriels en chaîne."""

    def test_chain_is_tree(self):
        """Une chaîne doit être un arbre (sans cycles)."""
        fg = generate_chain(5, seed=0)
        assert fg.is_tree()

    def test_chain_length(self):
        """La chaîne doit avoir le bon nombre de variables."""
        fg = generate_chain(8, seed=0)
        assert fg.n_variables == 8

    def test_chain_factor_count(self):
        """Une chaîne de longueur n doit avoir (n-1) facteurs par paires + n facteurs unaires."""
        n = 6
        fg = generate_chain(n, seed=0)
        assert fg.n_factors == (n - 1) + n
