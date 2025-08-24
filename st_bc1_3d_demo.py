#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# st_bc1_3d_demo.py  -  Spatial Transcriptomics demo (Bc1)
# Author: Jason Mercer
#
# What this does
# --------------
# 1) Loads a public Visium lymph-node dataset (.h5ad) directly from GitHub.
# 2) Selects gene 'Bc1' (fallback: highest-mean gene if Bc1 is absent).
# 3) Builds a kNN graph on tissue coordinates.
# 4) Computes smoothed expression (fast neighbor-mean; optional Laplacian).
# 5) Computes a simple stability score S = α·u01 − β·σ01 (coherence vs variation).
# 6) Generates 3D plots: raw, smoothed, stability (PNG).
# 7) Writes metrics.csv with basic stats (variance, reduction, spots).
#
# Dependencies
# ------------
# pip install:
#     scanpy anndata numpy scipy scikit-learn matplotlib
#
# Usage
# -----
#     python st_bc1_3d_demo.py
# (Outputs go to ./artifacts/)
#

import os
import json
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Tuple

# -----------------------
# Config (edit if needed)
# -----------------------
DATA_URL = (
    "https://github.com/romain-lopez/DestVI-reproducibility/raw/master/"
    "lymph_node/deconvolution/ST-LN-compressed.h5ad"
)
PREFERRED_GENE = "Bc1"
K_NEIGHBORS = 8             # spatial kNN for smoothing
USE_LAPLACIAN = False       # False = fast neighbor mean; True = graph Laplacian
ETA = 3.0                   # Laplacian solver strength if USE_LAPLACIAN=True
ALPHA, BETA = 1.0, 0.9      # stability weights
OUTDIR = "artifacts"


# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def minmax01(v: np.ndarray) -> np.ndarray:
    v = v.astype(float)
    vmin, vmax = float(v.min()), float(v.max())
    return (v - vmin) / (vmax - vmin + 1e-12)


def build_knn(coords: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    dists, idxs = nbrs.kneighbors(coords)  # idxs[i,0] == i (self)
    return dists, idxs


def smooth_knn_mean(raw: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    # mean over neighbors (excluding self)
    return np.array([raw[nbrs_i[1:]].mean() for nbrs_i in idxs])


def smooth_laplacian(coords: np.ndarray, raw: np.ndarray, dists: np.ndarray, idxs: np.ndarray,
                     eta: float) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    N = coords.shape[0]
    sigma = np.median(dists[:, 1:]) + 1e-9
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in idxs[i, 1:]:
            w = np.exp(-((coords[i] - coords[j]) ** 2).sum() / (sigma ** 2))
            rows.append(i); cols.append(j); vals.append(w)
    W = sp.coo_matrix((vals, (rows, cols)), shape=(N, N))
    W = 0.5 * (W + W.T)   # symmetrize
    D = sp.diags(W.sum(axis=1).A.ravel())
    L = D - W
    u = spsolve((L + eta * sp.eye(N)).tocsr(), eta * raw)
    return u


def local_variation(smoothed: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    # σ_i = sqrt( mean_j (u_i - u_j)^2 )
    diff2_mean = np.array([
        np.mean((smoothed[i] - smoothed[idxs[i, 1:]]) ** 2)
        for i in range(len(smoothed))
    ])
    return np.sqrt(diff2_mean)


def plot3d(coords: np.ndarray, values: np.ndarray, title: str, cmap: str,
           zlabel: str, outpath: str):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(coords[:, 0], coords[:, 1], values, c=values, cmap=cmap, s=10)
    ax.set_title(title, pad=16)
    ax.set_xlabel("X (tissue)")
    ax.set_ylabel("Y (tissue)")
    ax.set_zlabel(zlabel)
    cb = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label(zlabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(OUTDIR)

    # Load dataset
    adata = sc.read(DATA_URL)
    coords = np.array(adata.obsm["spatial"], float)  # (N,2)
    N = coords.shape[0]

    # Pick gene
    varnames = np.array(adata.var_names.astype(str))
    if PREFERRED_GENE in varnames:
        gene = PREFERRED_GENE
    else:
        X = adata.X
        mean_by_gene = (X.mean(axis=0).A.ravel() if hasattr(X, "A")
                        else np.array(X).mean(axis=0))
        gene = str(varnames[np.argmax(mean_by_gene)])
        print(f"[WARN] '{PREFERRED_GENE}' not found; using top-mean gene: {gene}")

    x = adata[:, gene].X
    raw = (x.A.ravel() if hasattr(x, "A") else np.asarray(x).ravel()).astype(float)

    # kNN graph
    dists, idxs = build_knn(coords, K_NEIGHBORS)

    # Smoothing
    if USE_LAPLACIAN:
        smoothed = smooth_laplacian(coords, raw, dists, idxs, ETA)
        smooth_kind = f"laplacian(eta={ETA})"
    else:
        smoothed = smooth_knn_mean(raw, idxs)
        smooth_kind = f"knn_mean(k={K_NEIGHBORS})"

    # Stability
    sigma_loc = local_variation(smoothed, idxs)
    u01, s01 = minmax01(smoothed), minmax01(sigma_loc)
    S = ALPHA * u01 - BETA * s01

    # Stats
    var_raw = float(np.var(raw))
    var_smooth = float(np.var(smoothed))
    reduction = var_raw / (var_smooth + 1e-12)

    print(f"[INFO] Gene={gene}  Spots={N}")
    print(f"[INFO] Smoothing={smooth_kind}")
    print(f"[INFO] var(raw)={var_raw:.5e}  var(smooth)={var_smooth:.5e}  reduction≈{reduction:.2e}")

    # Plots
    raw_png = os.path.join(OUTDIR, "bc1_3d_raw.png")
    smooth_png = os.path.join(OUTDIR, "bc1_3d_smooth.png")
    stab_png = os.path.join(OUTDIR, "bc1_3d_stability.png")

    plot3d(coords, raw,
           title=f"3D Raw Spatial Expression ({gene}) — spots={N}, var={var_raw:.2e}",
           cmap="viridis", zlabel="raw counts", outpath=raw_png)
    plot3d(coords, smoothed,
           title=f"3D Smoothed Spatial Expression ({gene}) — var={var_smooth:.2e}, reduction≈{reduction:.2e}",
           cmap="viridis", zlabel="smoothed", outpath=smooth_png)
    plot3d(coords, S,
           title=f"3D Stability Map ({gene}) — S = α·u01 − β·σ01  (α={ALPHA}, β={BETA})",
           cmap="coolwarm", zlabel="stability S", outpath=stab_png)

    # Save metrics
    metrics = {
        "gene": gene,
        "spots": int(N),
        "smoothing": smooth_kind,
        "variance_raw": var_raw,
        "variance_smooth": var_smooth,
        "reduction_factor": reduction,
        "alpha": ALPHA,
        "beta": BETA,
        "k_neighbors": K_NEIGHBORS,
        "laplacian_eta": ETA if USE_LAPLACIAN else None,
        "data_url": DATA_URL,
    }
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(OUTDIR, "metrics.csv"), "w") as f:
        f.write("gene,spots,smoothing,variance_raw,variance_smooth,reduction_factor,alpha,beta,k_neighbors,laplacian_eta\n")
        f.write(f"{gene},{N},{smooth_kind},{var_raw:.6e},{var_smooth:.6e},{reduction:.6e},{ALPHA},{BETA},{K_NEIGHBORS},{ETA if USE_LAPLACIAN else ''}\n")

    print(f"[DONE] Wrote plots to: {OUTDIR}/  and metrics.json / metrics.csv")


if __name__ == "__main__":
    main()
