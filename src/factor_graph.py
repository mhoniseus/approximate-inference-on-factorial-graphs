"""
Structures de données de graphes factoriels pour les modèles graphiques discrets.

Implémente les classes Factor et FactorGraph qui représentent des distributions
de probabilité discrètes sous forme de graphes bipartis de nœuds variables
et de nœuds facteurs.

Auteur : Mouhssine Rifaki
"""

import numpy as np
import networkx as nx
from itertools import product as cartesian_product


class Factor:
    """Un facteur (fonction de potentiel) dans un graphe factoriel.

    Un facteur associe des assignations de ses variables à des valeurs réelles
    non négatives. En interne, le potentiel est stocké sous forme de table dense
    indexée par le produit cartésien des cardinalités des variables.

    Paramètres
    ----------
    variables : list of str
        Noms des variables dont dépend ce facteur.
    cardinalities : list of int
        Nombre d'états pour chaque variable (même ordre que `variables`).
    values : np.ndarray ou None
        Table de potentiel de forme = tuple(cardinalities).
        Si None, initialisée à des uns (facteur uniforme).
    name : str ou None
        Étiquette optionnelle pour le facteur.
    """

    def __init__(self, variables, cardinalities, values=None, name=None):
        assert len(variables) == len(cardinalities)
        self.variables = list(variables)
        self.cardinalities = list(cardinalities)
        self.name = name or f"f({','.join(variables)})"

        shape = tuple(cardinalities)
        if values is not None:
            values = np.asarray(values, dtype=np.float64)
            assert values.shape == shape, (
                f"Forme attendue {shape}, obtenue {values.shape}"
            )
            self.values = values
        else:
            self.values = np.ones(shape, dtype=np.float64)

    @property
    def scope(self):
        """Ensemble des noms de variables couvertes par ce facteur."""
        return set(self.variables)

    def get_axis(self, variable):
        """Retourne l'indice de l'axe pour un nom de variable donné."""
        return self.variables.index(variable)

    def marginalize(self, variable):
        """Somme sur une seule variable, retournant un nouveau Factor.

        Paramètres
        ----------
        variable : str
            La variable sur laquelle marginaliser.

        Retourne
        -------
        Factor
            Nouveau facteur avec `variable` retirée de la portée.
        """
        axis = self.get_axis(variable)
        new_vars = [v for v in self.variables if v != variable]
        new_cards = [c for v, c in zip(self.variables, self.cardinalities) if v != variable]
        new_values = self.values.sum(axis=axis)
        return Factor(new_vars, new_cards, new_values)

    def maximize(self, variable):
        """Prend le maximum sur une seule variable, retournant un nouveau Factor.

        Paramètres
        ----------
        variable : str
            La variable sur laquelle maximiser.

        Retourne
        -------
        Factor
            Nouveau facteur avec `variable` retirée de la portée.
        """
        axis = self.get_axis(variable)
        new_vars = [v for v in self.variables if v != variable]
        new_cards = [c for v, c in zip(self.variables, self.cardinalities) if v != variable]
        new_values = self.values.max(axis=axis)
        return Factor(new_vars, new_cards, new_values)

    def multiply(self, other):
        """Multiplication point par point de deux facteurs, alignés sur les variables partagées.

        Paramètres
        ----------
        other : Factor
            Le facteur avec lequel multiplier.

        Retourne
        -------
        Factor
            Facteur produit dont la portée est l'union des deux portées.
        """
        # Construire la liste combinée des variables
        combined_vars = list(self.variables)
        combined_cards = list(self.cardinalities)
        for v, c in zip(other.variables, other.cardinalities):
            if v not in combined_vars:
                combined_vars.append(v)
                combined_cards.append(c)

        # Redimensionner self.values dans l'espace combiné
        shape_self = []
        for v in combined_vars:
            if v in self.variables:
                shape_self.append(self.cardinalities[self.variables.index(v)])
            else:
                shape_self.append(1)

        shape_other = []
        for v in combined_vars:
            if v in other.variables:
                shape_other.append(other.cardinalities[other.variables.index(v)])
            else:
                shape_other.append(1)

        # Approche plus simple : étendre les deux dans la forme combinée via la diffusion
        val_self = self.values.copy()
        val_other = other.values.copy()

        # Transposer self.values pour que les axes correspondent à l'ordre de combined_vars
        axes_self = [self.variables.index(v) for v in combined_vars if v in self.variables]
        val_self = np.transpose(val_self, axes_self)
        # Insérer des axes de longueur 1 pour les variables absentes de self
        insert_positions = [i for i, v in enumerate(combined_vars) if v not in self.variables]
        for pos in insert_positions:
            val_self = np.expand_dims(val_self, axis=pos)

        # Transposer other.values de la même manière
        axes_other = [other.variables.index(v) for v in combined_vars if v in other.variables]
        val_other = np.transpose(val_other, axes_other)
        insert_positions = [i for i, v in enumerate(combined_vars) if v not in other.variables]
        for pos in insert_positions:
            val_other = np.expand_dims(val_other, axis=pos)

        result = val_self * val_other
        return Factor(combined_vars, combined_cards, result)

    def normalize(self):
        """Normalise la table du facteur pour qu'elle somme à 1 (sur place)."""
        total = self.values.sum()
        if total > 0:
            self.values /= total
        return self

    def __repr__(self):
        return (
            f"Factor(name={self.name!r}, vars={self.variables}, "
            f"cards={self.cardinalities})"
        )


class FactorGraph:
    """Un graphe factoriel : graphe biparti de nœuds variables et de nœuds facteurs.

    Paramètres
    ----------
    name : str
        Étiquette optionnelle pour le graphe.
    """

    def __init__(self, name="FactorGraph"):
        self.name = name
        self.variables = {}      # var_name -> cardinalité
        self.factors = []        # liste d'objets Factor
        self._var_to_factors = {}  # var_name -> liste d'indices de facteurs

    def add_variable(self, name, cardinality):
        """Enregistre un nœud variable.

        Paramètres
        ----------
        name : str
            Identifiant de la variable.
        cardinality : int
            Nombre d'états (par ex. 2 pour binaire).
        """
        self.variables[name] = cardinality
        if name not in self._var_to_factors:
            self._var_to_factors[name] = []

    def add_factor(self, factor):
        """Ajoute un nœud facteur et le connecte à ses variables.

        Paramètres
        ----------
        factor : Factor
            Le facteur à ajouter. Toutes les variables dans sa portée doivent
            déjà être enregistrées.

        Retourne
        -------
        int
            Indice du facteur ajouté.
        """
        for v in factor.variables:
            if v not in self.variables:
                raise ValueError(
                    f"Variable {v!r} non enregistrée. "
                    f"Appelez d'abord add_variable."
                )
        idx = len(self.factors)
        self.factors.append(factor)
        for v in factor.variables:
            self._var_to_factors[v].append(idx)
        return idx

    def neighbors_of_variable(self, var_name):
        """Retourne les indices des facteurs connectés à une variable."""
        return list(self._var_to_factors.get(var_name, []))

    def neighbors_of_factor(self, factor_idx):
        """Retourne les noms des variables connectées à un facteur."""
        return list(self.factors[factor_idx].variables)

    @property
    def n_variables(self):
        return len(self.variables)

    @property
    def n_factors(self):
        return len(self.factors)

    def to_networkx(self):
        """Convertit en un graphe biparti NetworkX pour la visualisation.

        Les nœuds variables reçoivent l'attribut bipartite=0.
        Les nœuds facteurs reçoivent l'attribut bipartite=1.
        """
        G = nx.Graph()
        for v in self.variables:
            G.add_node(v, bipartite=0, node_type="variable")
        for i, f in enumerate(self.factors):
            fname = f.name
            G.add_node(fname, bipartite=1, node_type="factor")
            for v in f.variables:
                G.add_edge(v, fname)
        return G

    def is_tree(self):
        """Vérifie si le graphe factoriel n'a pas de cycles (est un arbre)."""
        G = self.to_networkx()
        return nx.is_tree(G)

    def joint_distribution(self):
        """Calcule la distribution jointe complète en multipliant tous les facteurs.

        Ceci n'est faisable que pour les petits graphes (énumération en force brute).

        Retourne
        -------
        Factor
            Un seul facteur sur toutes les variables.
        """
        if not self.factors:
            raise ValueError("Aucun facteur dans le graphe.")
        result = self.factors[0]
        for f in self.factors[1:]:
            result = result.multiply(f)
        result.normalize()
        return result

    def __repr__(self):
        return (
            f"FactorGraph(name={self.name!r}, "
            f"vars={self.n_variables}, factors={self.n_factors})"
        )
