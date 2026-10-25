"""Frozen local-Mahalanobis, inverse-distance prototype voting (no rejection)."""
import numpy as np
import pandas as pd


def distances(values, centers, precisions):
    result = np.empty((len(values), len(centers)))
    for j, (center, precision) in enumerate(zip(centers, precisions)):
        delta = values - center
        result[:, j] = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, precision, delta), 0))
    return result


def assign(values, centers, precisions, labels, neighbors=5, chunk_size=20000):
    """Assign standardised feature vectors; prototype IDs are their array indices.

    A distance tie follows NumPy argsort order, matching the archived source.
    A class-vote tie follows first appearance among the selected prototypes.
    Recorded distance/ID belong to the nearest prototype OF THE WINNING CLASS.
    """
    values, centers, precisions = (np.asarray(x, dtype=float) for x in (values, centers, precisions))
    labels = np.asarray(labels)
    if (values.ndim != 2 or centers.ndim != 2 or not len(centers)
            or values.shape[1] != centers.shape[1]
            or precisions.shape != (len(centers), centers.shape[1], centers.shape[1])
            or labels.shape != (len(centers),) or not np.issubdtype(labels.dtype, np.integer)
            or not all(np.isfinite(x).all() for x in (values, centers, precisions))):
        raise ValueError("Invalid feature/prototype dimensions, labels or nonfinite values")
    if neighbors < 1 or int(neighbors) != neighbors or chunk_size < 1 or int(chunk_size) != chunk_size:
        raise ValueError("neighbors and chunk_size must be positive integers")
    rows = []
    for start in range(0, len(values), chunk_size):
        matrix = distances(values[start:start+chunk_size], centers, precisions)
        selected = np.argsort(matrix, axis=1)[:, :min(neighbors, len(centers))]
        for row, indices in enumerate(selected):
            d = matrix[row, indices]
            classes = labels[indices]
            weights = 1.0 / (d + 1e-8)
            votes = {}
            for category, weight in zip(classes, weights):
                votes[int(category)] = votes.get(int(category), 0.0) + float(weight)
            winner = max(votes, key=votes.get)
            member = np.flatnonzero(classes == winner)
            nearest = member[np.argmin(d[member])]
            rows.append((winner, int(indices[nearest]), float(d[nearest]),
                         votes[winner] / max(sum(votes.values()), 1e-8)))
    return pd.DataFrame(rows, columns=("source_label", "prototype_id", "prototype_distance",
                                       "prototype_vote_share"))


def regroup(labels, mapping):
    """Apply an externally supplied numeric class mapping AFTER source voting."""
    labels = np.asarray(labels, dtype=int)
    mapping = {int(k): int(v) for k, v in mapping.items()}
    if not set(labels).issubset(mapping):
        raise ValueError("Class mapping does not cover all assigned source labels")
    return np.array([mapping[int(label)] for label in labels], dtype=int)
