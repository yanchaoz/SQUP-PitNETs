"""Descriptive SG review-field summaries from separately supplied measurements."""
from pathlib import Path

import numpy as np
import pandas as pd

SIDES = ("reference", "prediction")
CLASSES = (1, 2, 3, 4, 5)
FEATURES = ("area_px", "perimeter_px", "feret_max_px", "major_axis_px",
            "aspect_ratio", "extent", "convexity", "axis_skewness")
PREFIX = "Supplementary_Table_S5_"


def fractions(weights):
    """Exclude C5 before calling; empty C1-C4 compositions are undefined."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all():
        raise ValueError("Expected four finite class weights")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Undefined or invalid C1-C4 composition")
    return weights / weights.sum()


def compare(reference, prediction):
    delta = fractions(prediction) - fractions(reference)
    return delta * 100, float(np.abs(delta).sum() / 2)


def summarise(frame, unit):
    metrics = [c for c in frame.columns if c.endswith("_tvd") or c.endswith("_delta_pp")]
    records = []
    for metric in metrics:
        values = frame[metric]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite review-field summary")
        records.append(dict(unit=unit, metric=metric, n=len(values),
                            median=float(values.median()), q25=float(values.quantile(.25)),
                            q75=float(values.quantile(.75)), p95=float(values.quantile(.95)),
                            minimum=float(values.min()), maximum=float(values.max())))
    return records


def analyse(inputs, output):
    counts = ["n_profiles", "matched_both_included_n", "unmatched_n", "counterpart_outside_inclusion_n"]
    if inputs.empty or inputs[["crop_id", "case_id"]].isna().any().any():
        raise ValueError("Missing review fields or identifiers")
    if not inputs.groupby("crop_id").case_id.nunique().eq(1).all():
        raise ValueError("Each crop must belong to exactly one case")
    if inputs.duplicated(["crop_id", "side", "display_class"]).any():
        raise ValueError("Duplicate class rows")
    if not inputs.side.isin(SIDES).all() or not inputs.display_class.isin(CLASSES).all():
        raise ValueError("Unexpected side or class")
    values = inputs[counts].to_numpy()
    if not (np.isfinite(values).all() and (values >= 0).all() and (values == values.round()).all()):
        raise ValueError("Counts must be finite nonnegative integers")
    if not np.isfinite(inputs.full_profile_area_px).all() or (inputs.full_profile_area_px < 0).any():
        raise ValueError("Invalid profile area")
    if not np.array_equal(inputs.n_profiles, inputs[counts[1:]].sum(axis=1)):
        raise ValueError("Matching-status counts do not close")
    if not inputs.groupby(["crop_id", "side"]).display_class.nunique().eq(5).all():
        raise ValueError("Every side must contain all five class rows, including zero rows")
    if not inputs.groupby("crop_id").side.nunique().eq(2).all():
        raise ValueError("Every field must contain both sides")
    matched = inputs.groupby(["crop_id", "side"]).matched_both_included_n.sum().unstack()
    if not matched.reference.eq(matched.prediction).all():
        raise ValueError("Both-included matched totals must agree between sides")
    records = []
    for (crop_id, case_id), group in inputs.groupby(["crop_id", "case_id"], sort=True):
        row = dict(crop_id=crop_id, case_id=case_id)
        for side in SIDES:
            g = group.loc[group.side.eq(side)].set_index("display_class").loc[list(CLASSES)]
            row[side + "_n_C1_C5"] = int(g.n_profiles.sum())
            row[side + "_n_C1_C4"] = int(g.loc[1:4, "n_profiles"].sum())
            row[side + "_n_C5"] = int(g.loc[5, "n_profiles"])
        for mode, column in (("count", "n_profiles"), ("area", "full_profile_area_px")):
            weights = [group.loc[group.side.eq(side)].set_index("display_class").loc[1:4, column]
                       for side in SIDES]
            delta, tvd = compare(*weights)
            row[mode + "_tvd"] = tvd
            for i, value in enumerate(delta, start=1):
                row[f"{mode}_C{i}_delta_pp"] = float(value)
                row[f"{mode}_C{i}_absolute_delta_pp"] = float(abs(value))
            for side, weight in zip(SIDES, weights):
                for i, value in enumerate(fractions(weight), start=1):
                    row[f"{side}_{mode}_C{i}_fraction"] = float(value)
        records.append(row)
    fields = pd.DataFrame(records)
    metrics = [c for c in fields if c.endswith("_tvd") or c.endswith("_delta_pp")]
    cases = fields.groupby("case_id")[metrics].median().reset_index()
    cases.insert(1, "n_fields", fields.groupby("case_id").size().to_numpy())
    summary = pd.DataFrame(summarise(fields, "review_field") + summarise(cases, "case_median"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields.to_csv(output / (PREFIX + "review_field_composition.csv"), index=False)
    cases.to_csv(output / (PREFIX + "review_case_composition.csv"), index=False)
    summary.to_csv(output / (PREFIX + "review_composition_summary.csv"), index=False)
    return fields, cases, summary


def measurement_summary(fields):
    """Aggregate archived signed field errors; do not redo instance matching."""
    rows = []
    for label, col in [("count", "signed_count_bias_pct"), ("occupied_area", "signed_occupied_area_error_pct")]:
        values = fields[col].abs()
        if values.empty or not np.isfinite(values).all():
            raise ValueError("Invalid per-field measurement errors")
        rows.append(dict(measurement=label, n_fields=len(values), median=values.median(),
                         q25=values.quantile(.25), q75=values.quantile(.75),
                         p95=values.quantile(.95), maximum=values.max()))
    return pd.DataFrame(rows)
