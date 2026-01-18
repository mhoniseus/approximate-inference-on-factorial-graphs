"""Modèles graphiques : inférence discrète et apprentissage."""

from .factor_graph import Factor, FactorGraph
from .belief_propagation import sum_product_bp, max_product_bp, loopy_bp
from .variational import mean_field_vi, compute_elbo
from .trw import trw_s, compute_mrf_energy, lp_relaxation_map, compute_trw_bound
from .grid_mrf import GridMRF, grid_loopy_bp, grid_trw_s, grid_mean_field
from .stereo import (load_middlebury, download_middlebury, compute_matching_cost,
                     build_stereo_mrf, compute_disparity_error, run_stereo_experiment)
from .utils import (generate_ising_grid, generate_random_fg, plot_beliefs,
                    plot_convergence, plot_stereo_results, plot_energy_comparison)
