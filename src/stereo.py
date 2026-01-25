"""
Pipeline de correspondance stéréo utilisant l'inférence MRF.

Charge les paires stéréo Middlebury, construit un MRF sur grille,
et compare les méthodes d'inférence (BP, TRW-S, champ moyen)
sur l'estimation de la carte de disparité.
"""

import os
import time
import urllib.request

import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

from .grid_mrf import GridMRF, grid_loopy_bp, grid_trw_s, grid_mean_field


# Métadonnées des jeux de données Middlebury
_BASE_URL_2001 = "https://vision.middlebury.edu/stereo/data/scenes2001/data"
_BASE_URL_2003 = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata"

MIDDLEBURY_DATASETS = {
    "tsukuba": {
        "files": {
            "scene1.row3.col3.ppm": f"{_BASE_URL_2001}/tsukuba/scene1.row3.col3.ppm",
            "scene1.row3.col5.ppm": f"{_BASE_URL_2001}/tsukuba/scene1.row3.col5.ppm",
            "truedisp.row3.col3.pgm": f"{_BASE_URL_2001}/tsukuba/truedisp.row3.col3.pgm",
        },
        "left": "scene1.row3.col3.ppm",
        "right": "scene1.row3.col5.ppm",
        "gt": "truedisp.row3.col3.pgm",
        "n_disparities": 30,
        "scale": 8,  # col3-col5 = 2-column baseline, so scale = 16/2 = 8
    },
    "venus": {
        "files": {
            "im2.ppm": f"{_BASE_URL_2001}/venus/im2.ppm",
            "im6.ppm": f"{_BASE_URL_2001}/venus/im6.ppm",
            "disp2.pgm": f"{_BASE_URL_2001}/venus/disp2.pgm",
        },
        "left": "im2.ppm",
        "right": "im6.ppm",
        "gt": "disp2.pgm",
        "n_disparities": 20,
        "scale": 8,
    },
    "teddy": {
        "files": {
            "im2.png": f"{_BASE_URL_2003}/teddy/im2.png",
            "im6.png": f"{_BASE_URL_2003}/teddy/im6.png",
            "disp2.png": f"{_BASE_URL_2003}/teddy/disp2.png",
        },
        "left": "im2.png",
        "right": "im6.png",
        "gt": "disp2.png",
        "n_disparities": 60,
        "scale": 4,  # 2003 dataset: quarter-pixel accuracy, divide by 4
        "format": "png",
    },
}


def download_middlebury(name, data_dir="data/middlebury"):
    """Télécharge un jeu de données Middlebury s'il n'est pas déjà présent.

    Paramètres
    ----------
    name : str
        Nom du jeu de données : "tsukuba", "venus" ou "teddy".
    data_dir : str
        Répertoire de destination.
    """
    if name not in MIDDLEBURY_DATASETS:
        raise ValueError(f"Jeu de données inconnu : {name}. Choix : {list(MIDDLEBURY_DATASETS.keys())}")

    info = MIDDLEBURY_DATASETS[name]
    dest_dir = os.path.join(data_dir, name)
    os.makedirs(dest_dir, exist_ok=True)

    for filename, url in info["files"].items():
        filepath = os.path.join(dest_dir, filename)
        if os.path.exists(filepath):
            continue
        print(f"Téléchargement de {filename}...")
        urllib.request.urlretrieve(url, filepath)

    print(f"Jeu de données {name} prêt dans {dest_dir}")


def _read_ppm(path):
    """Lit un fichier PPM (P6) et retourne un tableau uint8 (H, W, 3)."""
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        assert header == 'P6', f"Format PPM non supporté : {header}"
        # Ignorer les commentaires
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        W, H = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(H, W, 3)


def _read_pgm(path):
    """Lit un fichier PGM (P5) et retourne un tableau uint8 (H, W)."""
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        assert header == 'P5', f"Format PGM non supporté : {header}"
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        W, H = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(H, W)


def load_middlebury(name, data_dir="data/middlebury"):
    """Charge une paire stéréo Middlebury et la vérité terrain.

    Paramètres
    ----------
    name : str
        Nom du jeu de données : "tsukuba", "venus" ou "teddy".
    data_dir : str
        Répertoire des données.

    Retourne
    -------
    left : np.ndarray (H, W) float64 en niveaux de gris [0, 255]
    right : np.ndarray (H, W) float64 en niveaux de gris [0, 255]
    gt_disparity : np.ndarray (H, W) float64, disparité en pixels
    n_disparities : int
    """
    if name not in MIDDLEBURY_DATASETS:
        raise ValueError(f"Jeu de données inconnu : {name}")

    info = MIDDLEBURY_DATASETS[name]
    base = os.path.join(data_dir, name)

    left_path = os.path.join(base, info["left"])
    right_path = os.path.join(base, info["right"])
    gt_path = os.path.join(base, info["gt"])

    if not os.path.exists(left_path):
        download_middlebury(name, data_dir)

    # Charger les images selon le format
    fmt = info.get("format", "ppm")
    if fmt == "png":
        left_rgb = plt.imread(left_path)
        right_rgb = plt.imread(right_path)
        gt_raw_img = plt.imread(gt_path)
        # plt.imread renvoie [0,1] float pour PNG ou [0,255] uint8
        if left_rgb.dtype == np.float32 or left_rgb.dtype == np.float64:
            left_rgb = (left_rgb * 255).astype(np.uint8)
            right_rgb = (right_rgb * 255).astype(np.uint8)
        if gt_raw_img.dtype == np.float32 or gt_raw_img.dtype == np.float64:
            gt_raw = (gt_raw_img * 255).astype(np.float64)
        else:
            gt_raw = gt_raw_img.astype(np.float64)
    else:
        left_rgb = _read_ppm(left_path)
        right_rgb = _read_ppm(right_path)
        gt_raw = _read_pgm(gt_path).astype(np.float64)

    # Convertir en niveaux de gris
    if left_rgb.ndim == 3:
        left = np.mean(left_rgb.astype(np.float64), axis=2)
        right = np.mean(right_rgb.astype(np.float64), axis=2)
    else:
        left = left_rgb.astype(np.float64)
        right = right_rgb.astype(np.float64)

    # Charger la vérité terrain
    gt_disparity = gt_raw / info["scale"]

    return left, right, gt_disparity, info["n_disparities"]


def compute_matching_cost(left, right, n_disparities, window_size=5,
                          method="sad"):
    """Calcule le coût de correspondance pixel par pixel (potentiels unaires).

    Paramètres
    ----------
    left, right : np.ndarray (H, W) float64
        Paire stéréo en niveaux de gris.
    n_disparities : int
        Nombre de niveaux de disparité (d = 0, ..., n_disparities - 1).
    window_size : int
        Taille de la fenêtre de correspondance (doit être impair).
    method : str
        "sad" (Sum of Absolute Differences) ou "ssd" (Sum of Squared Differences).

    Retourne
    -------
    cost : np.ndarray (H, W, n_disparities) float64
    """
    H, W = left.shape
    cost = np.zeros((H, W, n_disparities), dtype=np.float64)

    for d in range(n_disparities):
        # Compute |left(i,j) - right(i,j-d)| for valid pixels
        if d == 0:
            diff = left - right
        else:
            # Fill invalid border pixels with a high penalty before filtering
            # so that uniform_filter doesn't leak zeros into valid region
            diff = np.full((H, W), np.nan)
            if d < W:
                diff[:, d:] = left[:, d:] - right[:, :-d]

        if method == "sad":
            pixel_cost = np.abs(diff)
        elif method == "ssd":
            pixel_cost = diff ** 2
        else:
            raise ValueError(f"Méthode inconnue : {method}")

        # Agrégation par fenêtre (only over valid pixels)
        if d > 0:
            valid = ~np.isnan(pixel_cost)
            pc_clean = np.where(valid, pixel_cost, 0.0)
            sum_vals = uniform_filter(pc_clean, size=window_size) * (window_size ** 2)
            sum_valid = uniform_filter(valid.astype(float), size=window_size) * (window_size ** 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                cost[:, :, d] = np.where(sum_valid > 0, sum_vals / sum_valid, 1e6)
        else:
            cost[:, :, d] = uniform_filter(pixel_cost, size=window_size)

    return cost


def build_stereo_mrf(left, right, n_disparities, window_size=5,
                     cost_method="sad", pairwise_type="truncated_linear",
                     pairwise_weight=20.0, pairwise_trunc=2.0):
    """Construit un GridMRF pour la correspondance stéréo.

    Paramètres
    ----------
    left, right : np.ndarray (H, W)
    n_disparities : int
    window_size : int
    cost_method : str
    pairwise_type : str
    pairwise_weight : float
    pairwise_trunc : float

    Retourne
    -------
    mrf : GridMRF
    """
    cost = compute_matching_cost(left, right, n_disparities, window_size, cost_method)
    H, W = left.shape
    return GridMRF(H, W, n_disparities, cost,
                   pairwise_type=pairwise_type,
                   pairwise_weight=pairwise_weight,
                   pairwise_trunc=pairwise_trunc)


def compute_disparity_error(estimated, ground_truth, threshold=1.0):
    """Calcule les métriques d'erreur de disparité.

    Paramètres
    ----------
    estimated : np.ndarray (H, W) float ou int
    ground_truth : np.ndarray (H, W) float
    threshold : float
        Seuil pour la classification "mauvais pixel".

    Retourne
    -------
    bad_pixel_rate : float
        Fraction de pixels où |estimé - vérité| > seuil.
    mae : float
        Erreur absolue moyenne.
    rmse : float
        Erreur quadratique moyenne.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    # Masque des pixels valides (disparité > 0 dans la vérité terrain)
    valid = ground_truth > 0
    if not np.any(valid):
        return 0.0, 0.0, 0.0

    diff = np.abs(estimated[valid] - ground_truth[valid])
    bad_pixel_rate = np.mean(diff > threshold)
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    return bad_pixel_rate, mae, rmse


def run_stereo_experiment(name, data_dir="data/middlebury",
                          methods=None, pairwise_weight=20.0,
                          pairwise_trunc=2.0, max_iter=50,
                          window_size=5):
    """Exécute toutes les méthodes d'inférence sur une paire stéréo.

    Paramètres
    ----------
    name : str
        Nom du jeu de données Middlebury.
    data_dir : str
    methods : list of str ou None
        Méthodes à exécuter. Par défaut : ["bp", "trws", "mf"].
    pairwise_weight : float
    pairwise_trunc : float
    max_iter : int
    window_size : int

    Retourne
    -------
    left : np.ndarray (H, W)
    gt_disparity : np.ndarray (H, W)
    results : dict
        méthode -> {labeling, energy, bad_pixel_rate, mae, rmse, time_s, ...}
    """
    if methods is None:
        methods = ["bp", "trws", "mf"]

    left, right, gt_disparity, n_disp = load_middlebury(name, data_dir)
    mrf = build_stereo_mrf(left, right, n_disp, window_size=window_size,
                           pairwise_type="truncated_linear",
                           pairwise_weight=pairwise_weight,
                           pairwise_trunc=pairwise_trunc)

    results = {}

    for method in methods:
        print(f"  Exécution de {method} sur {name}...")
        t0 = time.time()

        if method == "bp":
            labeling, beliefs, energy_hist = grid_loopy_bp(
                mrf, max_iter=max_iter, damping=0.5
            )
            result = {"energy_history": energy_hist}

        elif method == "trws":
            labeling, beliefs, bound_hist, energy_hist = grid_trw_s(
                mrf, max_iter=max_iter
            )
            result = {"energy_history": energy_hist, "bound_history": bound_hist}

        elif method == "mf":
            labeling, q, energy_hist = grid_mean_field(
                mrf, max_iter=max_iter, seed=42
            )
            result = {"energy_history": energy_hist}

        else:
            raise ValueError(f"Méthode inconnue : {method}")

        elapsed = time.time() - t0
        energy = mrf.compute_energy(labeling)
        bpr, mae, rmse = compute_disparity_error(
            labeling.astype(float), gt_disparity
        )

        result.update({
            "labeling": labeling,
            "energy": energy,
            "bad_pixel_rate": bpr,
            "mae": mae,
            "rmse": rmse,
            "time_s": elapsed,
        })
        results[method] = result

        print(f"    {method}: énergie={energy:.0f}, "
              f"mauvais pixels={bpr:.1%}, MAE={mae:.2f}, "
              f"temps={elapsed:.1f}s")

    return left, gt_disparity, results
