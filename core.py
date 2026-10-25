"""Core algorithms; externally supplied images, feature tables and frozen atlas."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import assignment, atlas, features, reconstruction


def read_image(path):
    suffix = str(path).lower()
    if suffix.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
    elif suffix.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        array = np.asanyarray(nib.load(path).dataobj)
    elif suffix.endswith((".tif", ".tiff")):
        import tifffile
        array = tifffile.imread(path)
    else:
        raise ValueError("Expected .npy, .nii[.gz], or .tif[f]")
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError("Only 2-D fields are supported")
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    instances = sub.add_parser("instances", help="Boundary-aware watershed from binary predictions")
    instances.add_argument("--foreground", type=Path, required=True)
    instances.add_argument("--boundary", type=Path, required=True)
    instances.add_argument("--opening-radius", type=int, default=1)
    instances.add_argument("--closing-radius", type=int, default=1)
    instances.add_argument("--no-fill-holes", action="store_true")
    instances.add_argument("--min-seed-size", type=int, default=16)
    instances.add_argument("--min-instance-size", type=int, default=64)
    geometry = sub.add_parser("features", help="Eight features from one analysis tile, with explicit QC")
    geometry.add_argument("--labels", type=Path, required=True)
    geometry.add_argument("--min-area", type=float, default=50)
    geometry.add_argument("--mad-threshold", type=float, default=10.0)
    geometry.add_argument("--max-bad-features", type=int, default=7)
    geometry.add_argument("--no-mad-filter", action="store_true")
    geometry.add_argument("--case-id", default="")
    geometry.add_argument("--tile-name", default="")
    fitting = sub.add_parser("fit", help="Fit a new reference; not a replacement for a frozen study atlas")
    fitting.add_argument("--input-csv", type=Path, required=True)
    fitting.add_argument("--clusters", type=int, default=6)
    fitting.add_argument("--runs", type=int, default=20)
    fitting.add_argument("--seed", type=int, default=42)
    fitting.add_argument("--n-init", type=int, default=20)
    fitting.add_argument("--max-iter", type=int, default=300)
    fitting.add_argument("--prototypes", type=int, default=3)
    fitting.add_argument("--min-members", type=int, default=10)
    transfer = sub.add_parser("assign", help="Transfer eligible profiles to an external frozen atlas")
    transfer.add_argument("--input-csv", type=Path, required=True)
    transfer.add_argument("--atlas", type=Path, required=True)
    transfer.add_argument("--neighbors", type=int, default=5)
    transfer.add_argument("--chunk-size", type=int, default=20000)
    transfer.add_argument("--class-map", type=Path, help="Optional JSON: numeric source labels to numeric display labels")
    for command in (instances, geometry, fitting, transfer):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    if output == checkout or checkout in output.parents:
        parser.error("Write research outputs outside this code-only checkout")
    inputs = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    for path in inputs:
        parent = path.parent
        if parent == output or parent in output.parents or output in parent.parents:
            parser.error("Input and output directories must be separate and non-nested")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty to avoid overwriting prior results")
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "instances":
        labels = reconstruction.reconstruct(read_image(args.foreground), read_image(args.boundary),
                    args.opening_radius, args.closing_radius, not args.no_fill_holes,
                    args.min_seed_size, args.min_instance_size)
        np.save(output / "instances.npy", labels, allow_pickle=False)
    elif args.command == "features":
        frame = features.extract_features(read_image(args.labels), exclude_border=False)
        frame["case_id"], frame["tile_name"] = args.case_id, args.tile_name
        keep, reasons = features.quality_filter(frame, args.min_area, True,
                            None if args.no_mad_filter else args.mad_threshold, args.max_bad_features)
        frame.loc[keep].to_csv(output / "features.csv", index=False)
        rejected = frame.loc[~keep].copy()
        rejected["filter_reasons"] = np.array(reasons, dtype=str)[~keep]
        rejected.to_csv(output / "excluded_features.csv", index=False)
    else:
        frame = pd.read_csv(args.input_csv, dtype={"case_id": str, "tile_name": str}, float_precision="round_trip")
        if args.command == "fit":
            model, result = atlas.fit(frame, args.clusters, args.runs, args.seed, args.n_init,
                         args.max_iter, args.prototypes, args.min_members)
            atlas.save(model, output / "atlas.npz")
        else:
            model = atlas.load(args.atlas)
            result = assignment.assign(atlas.transform(frame, model), model["centers"],
                         model["precisions"], model["labels"], args.neighbors, args.chunk_size)
            if args.class_map:
                mapping = json.loads(args.class_map.read_text(encoding="utf-8"))
                result["display_label"] = assignment.regroup(result.source_label, mapping)
        if set(frame.columns) & set(result.columns):
            parser.error("Input contains output assignment columns; supply unassigned feature rows")
        pd.concat([frame.reset_index(drop=True), result], axis=1).to_csv(output / "assignments.csv", index=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("Completed; outputs and run settings are outside the code checkout.")


if __name__ == "__main__":
    main()
