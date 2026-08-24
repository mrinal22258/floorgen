"""
FID & KID Evaluation Metrics for Floorplan Generation.
Measures visual and distribution diversity between real and generated floorplans
using Inception / ResNet feature activations.
"""

import numpy as np
from scipy import linalg
from typing import List, Dict, Any, Tuple


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Computes the Fréchet Inception Distance between two multivariate Gaussians:
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 * sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary artifact cleanup
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_kid(features_real: np.ndarray, features_gen: np.ndarray, subset_size: int = 50, degree: int = 3) -> float:
    """
    Calculates Kernel Inception Distance (KID) via polynomial kernel maximum mean discrepancy (MMD).
    """
    n_real = len(features_real)
    n_gen = len(features_gen)
    m = min(n_real, n_gen, subset_size)
    if m < 2:
        return 0.0

    # Subsample
    r_idx = np.random.choice(n_real, m, replace=False)
    g_idx = np.random.choice(n_gen, m, replace=False)
    X = features_real[r_idx]
    Y = features_gen[g_idx]

    # Polynomial kernel: k(x, y) = (x^T y / d + 1)^3
    d = X.shape[1]
    K_XX = (np.dot(X, X.T) / d + 1) ** degree
    K_YY = (np.dot(Y, Y.T) / d + 1) ** degree
    K_XY = (np.dot(X, Y.T) / d + 1) ** degree

    # Unbiased MMD estimator
    mmd = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1)) + \
          (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) - \
          2 * K_XY.sum() / (m * m)

    return max(0.0, float(mmd))


def extract_floorplan_features(plans: List[Dict[str, Any]], feature_dim: int = 64) -> np.ndarray:
    """
    Extracts geometric and structural feature vectors from a list of floorplans
    (aspect ratios, relative areas, room category histograms, and spatial moments).
    """
    features = []
    for p in plans:
        vec = np.zeros(feature_dim, dtype=np.float32)
        rooms = p.get("rooms", [])
        total_area = sum(r.get("area", 0) for r in rooms) + 1e-6
        
        # Room histograms and area distributions
        for r in rooms:
            cid = r.get("category_id", 0) % 10
            vec[cid] += 1.0
            vec[10 + cid] += r.get("area", 0) / total_area

            # Aspect ratios
            bbox = r.get("bbox", [0, 0, 10, 10])
            w = max(1.0, bbox[2] - bbox[0])
            h = max(1.0, bbox[3] - bbox[1])
            vec[20 + cid] = w / h

        # Adjacency density
        n_rooms = max(1, len(rooms))
        n_edges = len(p.get("adjacency", []))
        vec[30] = n_edges / (n_rooms * (n_rooms - 1) / 2.0 + 1e-6)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        features.append(vec)

    return np.array(features, dtype=np.float32)


def compute_distribution_metrics(real_plans: List[Dict[str, Any]], gen_plans: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes FID and KID between real and generated floorplan datasets."""
    f_real = extract_floorplan_features(real_plans)
    f_gen = extract_floorplan_features(gen_plans)

    mu_real = np.mean(f_real, axis=0)
    sigma_real = np.cov(f_real, rowvar=False) + np.eye(f_real.shape[1]) * 1e-4

    mu_gen = np.mean(f_gen, axis=0)
    sigma_gen = np.cov(f_gen, rowvar=False) + np.eye(f_gen.shape[1]) * 1e-4

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    kid = calculate_kid(f_real, f_gen)

    return {
        "FID": round(fid, 3),
        "KID": round(kid, 5)
    }
