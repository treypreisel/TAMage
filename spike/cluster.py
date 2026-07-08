#!/usr/bin/env python3
"""Spike stage 2: cluster bake-off over the embedded companies.

Runs a small grid of UMAP x HDBSCAN configs, scores each on structural metrics,
and writes the results table plus per-config labels so the winner can be chosen
partly by LLM-judged coherence (stage 2b).

Outputs:
  spike/bakeoff_results.csv        one row per config: metrics
  spike/bakeoff_labels.npz         labels + probabilities per config
  spike/umap2d.npy                 shared 2D projection for the map (fixed seed)
"""

import itertools
import os
import time
import warnings

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(HERE, "embeddings.npz")

UMAP_GRID = [
    {"n_neighbors": 15, "n_components": 5},
    {"n_neighbors": 30, "n_components": 5},
    {"n_neighbors": 30, "n_components": 15},
    {"n_neighbors": 50, "n_components": 10},
]
HDB_GRID = [
    {"min_cluster_size": 60},
    {"min_cluster_size": 150},
    {"min_cluster_size": 300},
]
SEED = 42


def main():
    d = np.load(EMB, allow_pickle=True)
    V = d["vectors"]
    n = len(V)
    assert n > 5000, f"embeddings.npz holds only {n} vectors — rerun embed.py without --limit"
    print(f"{n:,} vectors · {len(UMAP_GRID) * len(HDB_GRID)} configs")

    results, labels_store = [], {}
    reduced_cache = {}
    for ui, ucfg in enumerate(UMAP_GRID):
        t0 = time.time()
        reducer = umap.UMAP(metric="cosine", random_state=SEED,
                            min_dist=0.0, **ucfg)
        reduced_cache[ui] = reducer.fit_transform(V)
        np.save(os.path.join(HERE, f"reduced_u{ui}.npy"), reduced_cache[ui])
        print(f"umap[{ui}] {ucfg} in {time.time()-t0:.0f}s", flush=True)

    for ui, ucfg in enumerate(UMAP_GRID):
        X = reduced_cache[ui]
        for hcfg in HDB_GRID:
            t0 = time.time()
            cl = hdbscan.HDBSCAN(min_samples=10, metric="euclidean",
                                 gen_min_span_tree=True,
                                 prediction_data=True, **hcfg)
            labels = cl.fit_predict(X)
            k = len(set(labels)) - (1 if -1 in labels else 0)
            noise = float((labels == -1).mean())
            sizes = pd.Series(labels[labels >= 0]).value_counts()
            key = f"u{ui}_m{hcfg['min_cluster_size']}"
            # cross-space-valid quality: silhouette on the ORIGINAL 768-d space
            sil = None
            if k >= 2:
                mask = labels >= 0
                sil = round(float(silhouette_score(
                    V[mask], labels[mask], metric="cosine",
                    sample_size=min(10000, int(mask.sum())), random_state=0)), 4)
            results.append({
                "key": key, **ucfg, **hcfg,
                "clusters": k, "noise_pct": round(noise * 100, 1),
                "biggest_pct": round(sizes.max() / n * 100, 1) if k else 0,
                "median_size": int(sizes.median()) if k else 0,
                "silhouette_768d": sil,
                "relative_validity": round(float(cl.relative_validity_), 4),
                "secs": round(time.time() - t0, 1),
            })
            labels_store[key + "_labels"] = labels
            labels_store[key + "_probs"] = cl.probabilities_
            if k > 0:  # persist soft membership top-2 for later margin computation
                mv = hdbscan.all_points_membership_vectors(cl)
                if mv.ndim == 2:
                    top2 = np.argsort(mv, axis=1)[:, -2:][:, ::-1]
                    labels_store[key + "_top2_idx"] = top2.astype(np.int16)
                    labels_store[key + "_top2_prob"] = np.take_along_axis(
                        mv, top2, axis=1).astype(np.float32)
            print(results[-1], flush=True)

    pd.DataFrame(results).to_csv(os.path.join(HERE, "bakeoff_results.csv"),
                                 index=False)
    np.savez_compressed(os.path.join(HERE, "bakeoff_labels.npz"), **labels_store)

    # shared 2D projection for the visual map, independent of the winning config
    t0 = time.time()
    xy = umap.UMAP(metric="cosine", random_state=SEED, n_neighbors=30,
                   n_components=2, min_dist=0.1).fit_transform(V)
    np.save(os.path.join(HERE, "umap2d.npy"), xy)
    print(f"2d map projection in {time.time()-t0:.0f}s — done")


if __name__ == "__main__":
    main()
