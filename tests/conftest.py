"""Fixtures partagées pour les tests de modèles graphiques."""

import numpy as np
import pytest
from src.factor_graph import Factor, FactorGraph


@pytest.fixture
def rng():
    """Générateur aléatoire déterministe."""
    return np.random.default_rng(3407)


@pytest.fixture
def binary_chain():
    """Une chaîne binaire simple à 3 variables : x0, x1, x2.

    Les facteurs par paires encouragent l'accord (couplage = 1.0).
    Les facteurs unaires sont uniformes.
    """
    fg = FactorGraph(name="BinaryChain3")
    for i in range(3):
        fg.add_variable(f"x{i}", 2)

    pair = np.array([[np.exp(1.0), np.exp(-1.0)],
                     [np.exp(-1.0), np.exp(1.0)]])

    fg.add_factor(Factor(["x0", "x1"], [2, 2], pair.copy(), name="f01"))
    fg.add_factor(Factor(["x1", "x2"], [2, 2], pair.copy(), name="f12"))

    # Facteurs unaires uniformes
    for i in range(3):
        fg.add_factor(Factor([f"x{i}"], [2], np.ones(2), name=f"u{i}"))

    return fg


@pytest.fixture
def biased_chain():
    """Une chaîne à 3 variables où x0 est biaisée vers l'état 1.

    Cela permet de vérifier que la BP propage l'évidence à travers la chaîne.
    """
    fg = FactorGraph(name="BiasedChain3")
    for i in range(3):
        fg.add_variable(f"x{i}", 2)

    pair = np.array([[np.exp(1.0), np.exp(-1.0)],
                     [np.exp(-1.0), np.exp(1.0)]])

    fg.add_factor(Factor(["x0", "x1"], [2, 2], pair.copy(), name="f01"))
    fg.add_factor(Factor(["x1", "x2"], [2, 2], pair.copy(), name="f12"))

    # x0 biaisée vers l'état 1
    fg.add_factor(Factor(["x0"], [2], np.array([0.1, 0.9]), name="bias0"))
    fg.add_factor(Factor(["x1"], [2], np.ones(2), name="u1"))
    fg.add_factor(Factor(["x2"], [2], np.ones(2), name="u2"))

    return fg


@pytest.fixture
def single_factor():
    """Une seule variable ternaire avec un facteur unaire non uniforme."""
    fg = FactorGraph(name="SingleFactor")
    fg.add_variable("y", 3)
    fg.add_factor(Factor(["y"], [3], np.array([1.0, 3.0, 2.0]), name="fy"))
    return fg


@pytest.fixture
def ising_4x4():
    """Une petite grille d'Ising 4x4 pour les tests d'intégration."""
    from src.utils import generate_ising_grid
    fg, grid_vars = generate_ising_grid(4, 4, coupling=0.5, field=0.1, seed=42)
    return fg, grid_vars
