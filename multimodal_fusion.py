"""
Task 3: fuse visual + audio features, evaluate without labels.
Fusion: concatenate visual + audio features, z-score them, then PCA.
Evaluation (no ground truth to check against):
1. Cluster the fused features, check silhouette score 
Score Meanings+1: Perfect separation; groups are far apart and tight.0: Groups touch or overlap.-1: Bad grouping; points likely are in the wrong cluster.
-- any real structure?
2. Bootstrap-resample, re-cluster, compare via ARI -- is it stable?
3. Check feature values sit in plausible ranges (face validity).
4. Correlate visual vs audio -- do both modalities agree?
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Part A: Fusion
# ---------------------------------------------------------------------------

def fuse_features(visual_df: pd.DataFrame, audio_df: pd.DataFrame, video_key="video"):
    """Merge Task 1 + Task 2 video-level feature tables on the video
    identifier, standardize, and return (fused_raw_df, fused_scaled_array,
    feature_names)."""
    merged = pd.merge(visual_df, audio_df, on=video_key, suffixes=("_vis", "_aud"))

    feature_cols = [c for c in merged.columns if c != video_key and merged[c].dtype != object]
    X = merged[feature_cols].copy()

    # Simple missing-value handling: median-impute (robust to outliers,
    # avoids dropping whole videos just because e.g. detection_rate was low
    # for a few frames). Documented as a simplification.
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return merged, X_scaled, feature_cols


def reduce_dimensionality(X_scaled, n_components=0.90):
    """PCA to `n_components` (if <1, treated as the variance-retained
    fraction). Purely for downstream clustering/visualization -- not used
    to discard the interpretable feature table itself."""
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca


def top_pca_features(pca, feature_names, n_top=3):
    # which original features matter most for each PCA component
    out = {}
    for i, component in enumerate(pca.components_):
        top_idx = np.argsort(np.abs(component))[::-1][:n_top]
        out[f"PC{i+1}"] = [(feature_names[j], round(float(component[j]), 2)) for j in top_idx]
    return out


# ---------------------------------------------------------------------------
# Part B: Evaluation without ground-truth labels
# ---------------------------------------------------------------------------

def internal_validity(X, k_range=range(2, 5)):
    """KMeans + silhouette score across a small range of k. Returns the
    best (k, silhouette) pair and the full sweep for transparency."""
    results = {}
    for k in k_range:
        if len(X) <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        results[k] = score

    if not results:
        return {"best_k": None, "best_silhouette": np.nan, "sweep": results}

    best_k = max(results, key=results.get)
    return {"best_k": best_k, "best_silhouette": results[best_k], "sweep": results}


def stability_bootstrap(X, k, n_bootstrap=20, sample_frac=0.8, seed=42):
    """Bootstrap-resample rows, re-cluster, and compare cluster assignments
    (on the shared subset of videos) against the full-sample clustering via
    Adjusted Rand Index. Returns mean +/- std ARI across bootstrap runs."""
    rng = np.random.RandomState(seed)
    n = len(X)

    full_km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    full_labels = full_km.labels_

    aris = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=int(n * sample_frac), replace=False)
        sub_km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X[idx])
        ari = adjusted_rand_score(full_labels[idx], sub_km.labels_)
        aris.append(ari)

    return {"mean_ari": float(np.mean(aris)), "std_ari": float(np.std(aris)), "n_bootstrap": n_bootstrap}


def cross_modal_agreement(merged_df):
    """Correlate a visual-expressivity index (mean of the visual *_var
    columns) against an audio-expressivity index (mean of the audio *_var
    columns). A significant positive correlation is evidence the two
    independently-computed modalities agree on which videos are more
    'expressive' overall -- a label-free consistency check."""
    visual_var_cols = [c for c in merged_df.columns if c.endswith("_var")]
    audio_var_cols = [c for c in merged_df.columns if c in
                       ("pitch_var_hz", "energy_var_rms")]

    if not visual_var_cols or not audio_var_cols:
        return {"r": np.nan, "p_value": np.nan, "note": "insufficient variance columns found"}

    visual_idx = merged_df[visual_var_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    audio_idx = merged_df[audio_var_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    valid = visual_idx.notna() & audio_idx.notna()
    if valid.sum() < 3:
        return {"r": np.nan, "p_value": np.nan, "note": "too few valid videos for correlation"}

    r, p = pearsonr(visual_idx[valid], audio_idx[valid])
    return {"r": float(r), "p_value": float(p), "n": int(valid.sum())}


def face_validity_report(merged_df):
    """Prints a small human-readable sanity-check summary (ranges/means of
    key features) so a reviewer can eyeball whether values are in a
    plausible range -- e.g. EAR/MAR in [0, ~1], pitch in typical human
    speech range (~80-400 Hz), speech_ratio in [0, 1]."""
    checks = {
        "eye_openness_mean": (0.0, 1.0),
        "mouth_openness_mean": (0.0, 1.5),
        "pitch_mean_hz": (60.0, 500.0),
        "speech_ratio": (0.0, 1.0),
    }
    report = {}
    for col, (lo, hi) in checks.items():
        if col in merged_df.columns:
            vals = pd.to_numeric(merged_df[col], errors="coerce").dropna()
            in_range = ((vals >= lo) & (vals <= hi)).mean() if len(vals) else np.nan
            report[col] = {
                "observed_range": (float(vals.min()), float(vals.max())) if len(vals) else None,
                "expected_range": (lo, hi),
                "fraction_in_expected_range": float(in_range) if not np.isnan(in_range) else None,
            }
    return report


def run_full_evaluation(visual_df: pd.DataFrame, audio_df: pd.DataFrame):
    """Orchestrates fusion + all label-free evaluation checks and returns
    a single results dict, ready to print or dump to JSON for the
    submission writeup."""
    merged, X_scaled, feature_cols = fuse_features(visual_df, audio_df)
    X_reduced, pca = reduce_dimensionality(X_scaled)

    iv = internal_validity(X_reduced)
    stability = (
        stability_bootstrap(X_reduced, iv["best_k"])
        if iv["best_k"] is not None else {"note": "skipped, no valid clustering found"}
    )
    cross_modal = cross_modal_agreement(merged)
    face_validity = face_validity_report(merged)

    return {
        "n_videos": len(merged),
        "n_fused_features": len(feature_cols),
        "pca_components_kept": X_reduced.shape[1],
        "pca_variance_retained": float(np.sum(pca.explained_variance_ratio_)),
        "pca_top_features": top_pca_features(pca, feature_cols),
        "internal_validity": iv,
        "stability_bootstrap": stability,
        "cross_modal_agreement": cross_modal,
        "face_validity": face_validity,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fuse visual + audio feature CSVs and run label-free evaluation."
    )
    parser.add_argument("visual_csv", help="CSV of Task 1 video-level features")
    parser.add_argument("audio_csv", help="CSV of Task 2 video-level features")
    args = parser.parse_args()

    visual_df = pd.read_csv(args.visual_csv)
    audio_df = pd.read_csv(args.audio_csv)

    results = run_full_evaluation(visual_df, audio_df)
    import json
    print(json.dumps(results, indent=2, default=str))
