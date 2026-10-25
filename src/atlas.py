"""Consensus clustering and mathematical prototypes in an eight-feature space."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .features import FEATURES


def consensus(values, clusters=6, runs=20, seed=42, n_init=20, max_iter=300):
    if any(int(v) != v or v < 1 for v in (clusters, runs, n_init, max_iter)):
        raise ValueError("Clustering counts must be positive integers")
    if len(values) < clusters or not np.isfinite(values).all():
        raise ValueError("Insufficient or nonfinite reference features")
    fitted = []
    for run in range(runs):
        model = KMeans(n_clusters=clusters, n_init=n_init, max_iter=max_iter,
                       random_state=seed + run*17 + clusters*131).fit(values)
        if len(np.unique(model.labels_)) != clusters:
            raise ValueError("Reference has too few distinct feature vectors")
        fitted.append(model)
    reference = fitted[int(np.argmin([m.inertia_ for m in fitted]))]
    votes = np.zeros((len(values), clusters), dtype=np.int32)
    for model in fitted:
        cost = np.linalg.norm(reference.cluster_centers_[:, None] - model.cluster_centers_[None], axis=2)
        ref_ids, current_ids = linear_sum_assignment(cost)
        mapping = np.empty(clusters, dtype=int)
        mapping[current_ids] = ref_ids
        votes[np.arange(len(values)), mapping[model.labels_]] += 1
    labels = votes.argmax(axis=1)
    fallback = len(np.unique(labels)) < clusters
    if fallback:
        labels = reference.labels_.copy()
        votes[:] = 0
        votes[np.arange(len(values)), labels] = runs
    ids, counts = np.unique(labels, return_counts=True)
    ordering = ids[np.argsort(-counts)]
    mapping = np.empty(clusters, dtype=int)
    mapping[ordering] = np.arange(clusters)
    return mapping[labels], votes.max(axis=1)/float(runs), fallback


def precision_matrix(values):
    n, dimensions = values.shape
    if n < 2:
        return np.eye(dimensions)
    if n >= 8:
        try:
            precision = LedoitWolf().fit(values).precision_
            if np.isfinite(precision).all():
                return precision
        except (ValueError, np.linalg.LinAlgError):
            pass
    variance = np.var(values, axis=0, ddof=1)
    variance = np.where(~np.isfinite(variance) | (variance < 1e-6), 1.0, variance)
    return np.diag(1.0 / variance)


def prototypes(values, labels, per_cluster=3, seed=42, min_members=10):
    if any(int(v) != v or v < 1 for v in (per_cluster, min_members)):
        raise ValueError("Prototype counts must be positive integers")
    centers, precisions, source_labels = [], [], []
    for category in sorted(np.unique(labels)):
        members = values[labels == category]
        count = max(1, min(per_cluster, len(members)))
        if len(members) < count * min_members:
            count = max(1, min(len(members)//min_members, len(members)))
        groups = (np.zeros(len(members), dtype=int) if count == 1 else
                  KMeans(n_clusters=count, random_state=seed+int(category)*97,
                         n_init=20, max_iter=300).fit_predict(members))
        for group in sorted(np.unique(groups)):
            subset = members[groups == group]
            centers.append(subset.mean(axis=0))
            precisions.append(precision_matrix(subset))
            source_labels.append(int(category))
    return np.stack(centers), np.stack(precisions), np.array(source_labels, dtype=int)


def fit(frame, clusters=6, runs=20, seed=42, n_init=20, max_iter=300,
        per_cluster=3, min_members=10):
    """Fit eligible reference profiles only. Input order and preprocessing matter.

    Tile-wise QC belongs upstream; do not pool tiles before applying that filter.
    Cluster IDs are size-ranked numeric labels, not biological identities.
    """
    values = frame.loc[:, FEATURES].to_numpy(float)
    if not len(values) or np.isinf(values).any() or np.isnan(values).all(axis=0).any():
        raise ValueError("Empty reference, infinite values or wholly missing feature")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputer.fit_transform(values))
    labels, confidence, fallback = consensus(standardized, clusters, runs, seed, n_init, max_iter)
    centers, precisions, prototype_labels = prototypes(standardized, labels, per_cluster, seed, min_members)
    config = dict(clusters=clusters, runs=runs, seed=seed, n_init=n_init, max_iter=max_iter,
                  per_cluster=per_cluster, min_members=min_members, consensus_fallback=fallback)
    model = dict(features=np.array(FEATURES), median=imputer.statistics_, mean=scaler.mean_,
                 scale=scaler.scale_, centers=centers, precisions=precisions, labels=prototype_labels,
                 config=np.array(json.dumps(config)))
    assignments = pd.DataFrame(dict(source_label=labels, consensus_confidence=confidence))
    return model, assignments


def validate(model):
    if list(model["features"]) != list(FEATURES):
        raise ValueError("Model feature order does not match the eight-feature schema")
    dimensions = len(FEATURES)
    for name in ("median", "mean", "scale"):
        if model[name].shape != (dimensions,) or not np.isfinite(model[name]).all():
            raise ValueError("Invalid scaler/imputer values")
    centers = model["centers"]
    if (centers.ndim != 2 or centers.shape[1] != dimensions or not len(centers)
            or model["precisions"].shape != (len(centers), dimensions, dimensions)
            or model["labels"].shape != (len(centers),)
            or not np.issubdtype(model["labels"].dtype, np.integer)
            or np.any(model["scale"] <= 0)
            or not np.isfinite(centers).all() or not np.isfinite(model["precisions"]).all()):
        raise ValueError("Invalid prototype representation")
    for precision in model["precisions"]:
        if not np.allclose(precision, precision.T) or np.linalg.eigvalsh(precision).min() < -1e-8:
            raise ValueError("Precision must be symmetric positive semidefinite")


def transform(frame, model):
    validate(model)
    values = frame.loc[:, list(model["features"])].to_numpy(float)
    if np.isinf(values).any():
        raise ValueError("Infinite application features are invalid")
    return (np.where(np.isnan(values), model["median"], values) - model["mean"]) / model["scale"]


def save(model, path):
    validate(model)
    # Numeric/string arrays only: never serialise executable pickle objects.
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **model)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        model = {key: archive[key] for key in archive.files}
    validate(model)
    return model
