"""The eight archived geometry descriptors, plus object-location metadata."""
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, convex_hull_image, disk

FEATURES = ("aspect_ratio", "extent", "convexity", "axis_skewness", "area_px",
            "perimeter_px", "feret_max_px", "major_axis_px")
LOCATION = ("instance_id", "centroid_row", "centroid_col", "bbox_min_row",
            "bbox_min_col", "bbox_max_row", "bbox_max_col", "touches_border")


def axis_descriptors(mask):
    coords = np.column_stack(np.nonzero(mask)).astype(float)
    if len(coords) < 2:
        return np.nan, np.nan
    centered = coords - coords.mean(axis=0, keepdims=True)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    projections = centered @ vectors[:, np.argsort(values)[::-1]]
    major, minor = projections.T
    span = float(np.ptp(major))
    if len(coords) < 5:
        return span, np.nan
    bins = np.linspace(major.min(), major.max(), max(10, int(np.ceil(span))) + 1)
    indices = np.clip(np.digitize(major, bins) - 1, 0, len(bins) - 2)
    widths = np.array([np.ptp(minor[indices == i]) if np.any(indices == i) else 0
                       for i in range(len(bins) - 1)])
    widths = widths[widths > 0]
    # This is skewness of nonzero transverse widths, not of pixel coordinates.
    return span, float(skew(widths)) if len(widths) >= 8 else np.nan


def extract_features(labels, exclude_border=True):
    labels = np.asarray(labels)
    if (labels.ndim != 2 or 0 in labels.shape or not np.isfinite(labels).all()
            or np.any(labels < 0) or not np.equal(labels, np.floor(labels)).all()):
        raise ValueError("Expected a 2-D nonnegative integer instance label map")
    labels = labels.astype(np.int64)
    rows = []
    for region in regionprops(labels):
        r0, c0, r1, c1 = region.bbox
        border = r0 == 0 or c0 == 0 or r1 == labels.shape[0] or c1 == labels.shape[1]
        if exclude_border and border:
            continue
        # Match the archived padded mask used for hull-boundary pixel counting.
        mask = labels[max(0, r0-8):r1+8, max(0, c0-8):c1+8] == region.label
        hull = convex_hull_image(mask)
        hull_boundary = np.count_nonzero(hull & ~binary_erosion(hull, disk(1)))
        span, axis_skewness = axis_descriptors(mask)
        perimeter = float(region.perimeter)
        minor = float(region.axis_minor_length)
        rows.append(dict(instance_id=int(region.label), centroid_row=region.centroid[0],
                         centroid_col=region.centroid[1], bbox_min_row=r0, bbox_min_col=c0,
                         bbox_max_row=r1, bbox_max_col=c1, touches_border=bool(border),
                         area_px=float(region.area), perimeter_px=perimeter,
                         major_axis_px=float(region.axis_major_length),
                         aspect_ratio=region.axis_major_length/minor if minor > 1e-12 else np.nan,
                         extent=float(region.extent),
                         convexity=hull_boundary/perimeter if perimeter > 1e-12 else np.nan,
                         feret_max_px=span, axis_skewness=axis_skewness))
    return pd.DataFrame(rows, columns=LOCATION + FEATURES)


def quality_filter(frame, min_area=50, drop_nonfinite=True, mad_threshold=10.0,
                   max_bad_features=7):
    """Filter ONE tile, returning a Boolean keep mask and reasons for every row.

    MAD screening is per tile, not after pooling reference cases. Set
    mad_threshold=None to disable it. An outlier needs >=max_bad_features flagged
    dimensions, not a single feature reaching that many MADs.
    """
    values = frame.loc[:, FEATURES].to_numpy(float)
    reasons = [[] for _ in range(len(frame))]

    def flag(bad, reason):
        for index in np.flatnonzero(bad):
            reasons[index].append(reason)

    area = frame.area_px.to_numpy(float)
    flag(~np.isfinite(area) | (area < min_area), "small_area")
    if "touches_border" in frame:
        flag(frame.touches_border.to_numpy(bool), "edge_instance")
    if drop_nonfinite:
        flag(~np.isfinite(values).all(axis=1), "nonfinite_feature")
    if mad_threshold is not None:
        if mad_threshold <= 0 or not 1 <= max_bad_features <= len(FEATURES):
            raise ValueError("Invalid MAD settings")
        count = np.zeros(len(frame), dtype=int)
        for column in values.T:
            finite = np.isfinite(column)
            if finite.sum() < 5:
                continue
            median = np.median(column[finite])
            mad = np.median(np.abs(column[finite] - median))
            if mad >= 1e-12:
                count += finite & (np.abs(0.6745 * (column - median) / mad) > mad_threshold)
        flag(count >= max_bad_features, "mad_outlier")
    return np.array([not r for r in reasons], dtype=bool), [";".join(r) for r in reasons]
