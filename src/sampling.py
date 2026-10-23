"""Grid-based density and morphology-composition sampling.

All observations and specimen identifiers must be supplied externally.
The legacy option replays historical area-threshold arithmetic only; it does
not calculate new tissue areas. Corrected inputs are required for new analyses.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

OUT = None
REPEATS = 1000
AREA_MODE = "corrected"
DENSITY_SEED = 20260829
COMPOSITION_SEED = 20261009
CLASSES = ["C1", "C2", "C3", "C4"]


def rng_for(seed, *parts):
    digest = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(digest, "little") + seed) % (2**32 - 1))


def load_csv(path):
    return pd.read_csv(path, dtype={"case_id": str}, float_precision="round_trip")


def arrays(frame, columns, eligible_only=False):
    d = frame[frame.candidate_cell == 1] if eligible_only else frame
    if d.empty:
        raise ValueError("No eligible base grid cells")
    shape = (int(d.grid_i.max()) + 1, int(d.grid_j.max()) + 1)
    c = np.zeros(shape + ((len(columns),) if len(columns) > 1 else ()))
    a = np.zeros(shape)
    iy, ix = d.grid_i.to_numpy(int), d.grid_j.to_numpy(int)
    c[iy, ix] = d[columns].to_numpy() if len(columns) > 1 else d[columns[0]].to_numpy()
    a[iy, ix] = d.valid_area_um2.to_numpy()
    return c, a


def integral(a):
    return np.pad(a.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)) + ((0, 0),) * (a.ndim - 2))


def nm_areas(a):
    units = np.rint(a * 1e6).astype(np.int64)
    if np.max(np.abs(units / 1e6 - a)) >= 1e-7:
        raise ValueError("Area is not representable in square nanometres")
    return units


def density_windows(c, a):
    area = nm_areas(a) if AREA_MODE == "corrected" else a
    scale = 1e6 if AREA_MODE == "corrected" else 1
    groups = []
    for oi in range(2):
        for oj in range(2):
            cc, aa = [], []
            for i in range(oi, a.shape[0] - 1, 2):
                for j in range(oj, a.shape[1] - 1, 2):
                    size = area[i:i+2, j:j+2].sum()
                    if size >= 200 * scale:
                        cc.append(float(c[i:i+2, j:j+2].sum()))
                        aa.append(float(size / scale))
            if aa:
                groups.append((np.asarray(cc), np.asarray(aa)))
    return groups


def composition_windows(c, a, side, bounds):
    if side < 10 or side % 10:
        raise ValueError("Field side must be a positive multiple of 10 micrometres")
    k = side // 10
    ic = integral(c)
    ia = integral(nm_areas(a)) if AREA_MODE == "corrected" else integral(a)
    scale = 1e6 if AREA_MODE == "corrected" else 1
    h = min(a.shape[0], int(np.floor(bounds[0] / 10)))
    w = min(a.shape[1], int(np.floor(bounds[1] / 10)))
    groups = []
    for oi in range(k):
        for oj in range(k):
            ii, jj = np.meshgrid(np.arange(oi, h-k+1, k), np.arange(oj, w-k+1, k), indexing="ij")
            i, j = ii.ravel(), jj.ravel()
            area = ia[i+k, j+k] - ia[i, j+k] - ia[i+k, j] + ia[i, j]
            count = ic[i+k, j+k] - ic[i, j+k] - ic[i+k, j] + ic[i, j]
            keep = area >= .5 * side**2 * scale
            if keep.any():
                groups.append((count[keep], area[keep] / scale))
    return groups


def draw(groups, m, rng):
    eligible = [g for g in groups if len(g[1]) >= m]
    if not eligible:
        return None
    counts, areas = [], []
    for _ in range(REPEATS):
        c, a = eligible[int(rng.integers(len(eligible)))]
        ix = rng.choice(len(a), m, replace=False)
        counts.append(c[ix].sum(axis=0))
        areas.append(a[ix].sum())
    return np.asarray(counts), np.asarray(areas)


def density_analysis(cases, grids, domains):
    rows = []
    for case in cases:
        for org in ("SG", "Mito"):
            d = grids[(grids.case_id == case) & (grids.organelle == org)]
            c, a = arrays(d, ["count"], eligible_only=True)
            meta = domains[(domains.case_id == case) & (domains.organelle == org)].iloc[0]
            full = float(meta.full_density_per_um2)
            groups = density_windows(c, a)
            rng = rng_for(DENSITY_SEED, case, org, 20, "random")
            raw = {}
            for m in (1, 3, 5, 10, 20, 50, 100):
                result = draw(groups, m, rng)
                if result is None:
                    continue
                count, area = result
                estimate = count / area
                error = 100 * abs(estimate - full) / full
                rows.append(dict(case_id=case, organelle=org, field_um=20, m=m, n_repeats=REPEATS,
                                 ape_q25=np.quantile(error, .25), ape_median=np.median(error), ape_q75=np.quantile(error, .75),
                                 estimate_median=np.median(estimate), success_probability_10pct=np.mean(error <= 10),
                                 sampled_area_median_um2=np.median(area), full_density_per_um2=full))
                raw["m" + str(m) + "_error_pct"] = error
                raw["m" + str(m) + "_valid_area_um2"] = area
                raw["m" + str(m) + "_estimate_density"] = estimate
            np.savez_compressed(OUT / (case + "_" + org + "_density.npz"), **raw)
    if not rows:
        raise ValueError("No feasible density sampling conditions")
    table = pd.DataFrame(rows)
    cohort = []
    for (org, m), d in table.groupby(["organelle", "m"]):
        cohort.append(dict(organelle=org, field_um=20, m=m, metric="ape_median", n_cases=len(d),
                           q25=np.quantile(d.ape_median, .25), median=np.median(d.ape_median), q75=np.quantile(d.ape_median, .75)))
    table.to_csv(OUT / "figure4D_case_summary.csv", index=False)
    pd.DataFrame(cohort).to_csv(OUT / "figure4D_cohort.csv", index=False)
    return table, pd.DataFrame(cohort)


def sample_stats(case, side, m, c, a, full, normal):
    total = c.sum(axis=1)
    f = np.divide(c, total[:, None], out=np.zeros_like(c, dtype=float), where=total[:, None] > 0)
    tvd = .5 * np.abs(f - full).sum(axis=1)
    tvd[total == 0] = np.nan
    enriched = full > normal
    sample_set = (f > normal) & (total[:, None] > 0)
    indices = np.flatnonzero(enriched)
    secondary = int(indices[np.argsort(-full[indices])][1]) if len(indices) >= 2 else -1
    row = dict(case_id=case, field_side_um=side, m=m, nominal_area_um2=side**2*m,
               n_resamples=REPEATS, empty_sample_probability=np.mean(total == 0),
               category_set_recovery_probability=np.mean(np.all(sample_set == enriched, axis=1)),
               any_enriched_component_under_call_probability=np.mean(np.any(enriched & ~sample_set, axis=1)),
               secondary_class="C" + str(secondary + 1) if secondary >= 0 else "",
               secondary_under_call_probability=np.mean(~sample_set[:, secondary]) if secondary >= 0 else np.nan,
               secondary_absent_probability=np.mean(c[:, secondary] == 0) if secondary >= 0 else np.nan,
               tvd_le_005_probability=np.mean(np.isfinite(tvd) & (tvd <= .05)),
               tvd_le_002_probability=np.mean(np.isfinite(tvd) & (tvd <= .02)))
    for prefix, values in [("tvd", tvd), ("valid_area_um2", a), ("sample_objects", total)]:
        values = values[np.isfinite(values)]
        q = np.quantile(values, [.1, .25, .5, .75, .9, .95]) if len(values) else [np.nan] * 6
        row.update({prefix + "_" + n: v for n, v in zip(("q10", "q25", "median", "q75", "q90", "q95"), q)})
    np.savez_compressed(OUT / (case + "_composition_" + str(side) + "_" + str(m) + ".npz"),
                        tvd=tvd, valid_area_um2=a, sample_fraction=f,
                        secondary_under_call=~sample_set[:, secondary] if secondary >= 0 else np.array([]),
                        secondary_absent=c[:, secondary] == 0 if secondary >= 0 else np.array([]))
    return row


def composition_analysis(cases, grids, domains, reference_id):
    normal_counts = grids.loc[grids.case_id == reference_id, CLASSES].sum().to_numpy(float)
    if normal_counts.sum() <= 0:
        raise ValueError("The supplied reference has no C1-C4 counts")
    normal = normal_counts / normal_counts.sum()
    rows, inventory = [], []
    for case in cases:
        c, a = arrays(grids[grids.case_id == case], CLASSES)
        if c.sum() <= 0:
            raise ValueError("Undefined full-region composition")
        full = c.sum(axis=(0, 1)) / c.sum()
        meta = domains[(domains.case_id == case) & (domains.organelle == "SG")].iloc[0]
        bounds = (meta.height_um, meta.width_um)
        inventory.append(dict(case_id=case, n_objects=int(c.sum()), analysable_area_um2=a.sum(),
                              **{CLASSES[k] + "_fraction": full[k] for k in range(4)}))
        for side in (20, 50):
            groups = composition_windows(c, a, side, bounds)
            for m in ((1,) if side == 20 else (1, 2, 4, 8, 16, 32, 64)):
                parts = (case, m, "20um_composition") if side == 20 else (case, side, m, "composition")
                sampled = draw(groups, m, rng_for(COMPOSITION_SEED, *parts))
                if sampled is not None:
                    rows.append(sample_stats(case, side, m, sampled[0], sampled[1], full, normal))
    table = pd.DataFrame(rows)
    if table.empty or not (table.field_side_um == 50).any():
        raise ValueError("No feasible composition sampling conditions")
    table.to_csv(OUT / "figure5_sampling.csv", index=False)
    pd.DataFrame(inventory).to_csv(OUT / "linked_inventory.csv", index=False)
    required = []
    for case, d in table[table.field_side_um == 50].groupby("case_id"):
        d = d.sort_values("m")
        success = d.tvd_le_005_probability.to_numpy() >= .9
        sustained = np.logical_and.accumulate(success[::-1])[::-1]
        ok = d[sustained]
        r = ok.iloc[0] if len(ok) else None
        required.append(dict(case_id=case, required_m=float(r.m) if r is not None else np.nan,
                             required_nominal_area_um2=float(r.nominal_area_um2) if r is not None else np.nan,
                             required_valid_area_median_um2=float(r.valid_area_um2_median) if r is not None else np.nan,
                             maximum_tested_area_um2=d.nominal_area_um2.max()))
    required = pd.DataFrame(required)
    required.to_csv(OUT / "figure5E_required_coverage.csv", index=False)
    return table, required, pd.DataFrame(inventory)


def load_inputs(folder, reference_id):
    folder = Path(folder)
    cases = json.loads((folder / "case_order.json").read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases or len(cases) != len(set(cases)):
        raise ValueError("case_order.json must be a nonempty list of unique string identifiers")
    if any(not isinstance(c, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cases):
        raise ValueError("Identifiers must be filesystem-safe strings")
    if reference_id not in cases:
        raise ValueError("Reference identifier is absent from case_order.json")
    domains = load_csv(folder / "case_domains.csv")
    d = load_csv(folder / "density_grid.csv.gz")
    c = load_csv(folder / "composition_grid.csv.gz")
    if domains.duplicated(["case_id", "organelle"]).any():
        raise ValueError("Duplicate analysis domains")
    expected_keys = {(case, org) for case in cases for org in ("SG", "Mito")}
    if set(zip(domains.case_id, domains.organelle)) != expected_keys or set(zip(d.case_id, d.organelle)) != expected_keys or set(c.case_id) != set(cases):
        raise ValueError("Inconsistent identifiers or organelle branches")
    for frame, keys, columns in [(d, ["case_id", "organelle", "grid_i", "grid_j"], ["count"]),
                                  (c, ["case_id", "grid_i", "grid_j"], CLASSES)]:
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate grid cells")
        v = frame[columns + ["grid_i", "grid_j"]].to_numpy()
        if not (np.isfinite(v).all() and (v >= 0).all() and (v == v.round()).all()):
            raise ValueError("Grid counts and indices must be finite nonnegative integers")
        if not np.isfinite(frame.valid_area_um2).all() or (frame.valid_area_um2 < 0).any():
            raise ValueError("Invalid tissue area")
    if not d.candidate_cell.isin([0, 1]).all():
        raise ValueError("candidate_cell must be binary")
    for (case, org), group in d.groupby(["case_id", "organelle"]):
        row = domains[(domains.case_id == case) & (domains.organelle == org)].iloc[0]
        if not all(np.isfinite(row[k]) and row[k] > 0 for k in ("height_um", "width_um", "valid_area_um2", "full_density_per_um2")):
            raise ValueError("Nonpositive or nonfinite domain geometry or reference density")
        if group["count"].sum() != row.n_profiles or not np.isclose(group.valid_area_um2.sum(), row.valid_area_um2, atol=1e-9, rtol=1e-11):
            raise ValueError("Grid totals differ from the full analysis domain")
        if not np.isclose(row.n_profiles / row.valid_area_um2, row.full_density_per_um2, atol=1e-12, rtol=1e-11):
            raise ValueError("Full density differs from count / valid area")
        if AREA_MODE == "corrected":
            dy = np.maximum(0., np.minimum(group.y1_um, row.height_um) - group.y0_um)
            dx = np.maximum(0., np.minimum(group.x1_um, row.width_um) - group.x0_um)
            if (group.valid_area_um2 > dx * dy + 1e-7).any():
                raise ValueError("Valid area exceeds physical grid support")
    return cases, d, c, domains


def cohort_facts(composition, required, reference_id):
    tumour = composition[(composition.case_id != reference_id) & (composition.field_side_um == 50)]
    one = tumour[tumour.m == 1]
    secondary = composition[(composition.case_id != reference_id) & (composition.m == 1) & composition.secondary_under_call_probability.notna()]
    req = required[required.case_id != reference_id]
    return dict(cohort_median_TVD_1_field=float(one.tvd_median.median()),
                cases_with_median_TVD_le005=int((one.tvd_median <= .05).sum()),
                stable_at_one_field=int((req.required_m == 1).sum()),
                median_required_nominal_area_um2=float(req.required_nominal_area_um2.median()),
                feasible_tumours_by_m={str(m): len(g) for m, g in tumour.groupby("m")},
                secondary_risk={str(side): dict(n_cases=len(g), median_undercall_pct=100*float(g.secondary_under_call_probability.median()),
                                               median_absence_pct=100*float(g.secondary_absent_probability.median())) for side, g in secondary.groupby("field_side_um")})
