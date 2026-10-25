"""Code-only CLI: external research inputs and outputs outside the checkout."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import sampling, sg_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    spatial = sub.add_parser("spatial", help="Density and composition resampling")
    spatial.add_argument("--input-dir", type=Path, required=True)
    spatial.add_argument("--reference-id", required=True)
    spatial.add_argument("--area-mode", choices=("corrected", "legacy"), default="corrected")
    spatial.add_argument("--repeats", type=int, default=1000)
    spatial.add_argument("--density-seed", type=int, default=sampling.DENSITY_SEED)
    spatial.add_argument("--composition-seed", type=int, default=sampling.COMPOSITION_SEED)
    spatial.add_argument("--output-dir", type=Path, required=True)
    sg = sub.add_parser("sg", help="SG review-field composition and optional measurement summaries")
    sg.add_argument("--input-csv", type=Path, required=True)
    sg.add_argument("--field-measurements", type=Path)
    sg.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parent
    if output == checkout or checkout in output.parents:
        parser.error("Write research outputs outside this code-only checkout")
    source = args.input_dir.resolve() if args.command == "spatial" else args.input_csv.resolve().parent
    if source == output or source in output.parents or output in source.parents:
        parser.error("Input and output locations must be separate, non-nested directories")
    if args.command == "spatial":
        if args.repeats < 1:
            parser.error("--repeats must be positive")
        sampling.REPEATS = args.repeats
        sampling.AREA_MODE = args.area_mode
        sampling.DENSITY_SEED = args.density_seed
        sampling.COMPOSITION_SEED = args.composition_seed
        cases, density, composition, domains = sampling.load_inputs(source, args.reference_id)
        output.mkdir(parents=True, exist_ok=True)
        sampling.OUT = output
        sampling.density_analysis(cases, density, domains)
        table, required, inventory = sampling.composition_analysis(cases, composition, domains, args.reference_id)
        facts = sampling.cohort_facts(table, required, args.reference_id)
        facts.update(area_mode=args.area_mode, repeats=args.repeats, density_seed=args.density_seed,
                     composition_seed=args.composition_seed, reference_id=args.reference_id,
                     linked_profiles=int(inventory.n_objects.sum()))
        (output / "summary.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    else:
        inputs = pd.read_csv(args.input_csv, dtype={"case_id": str, "crop_id": str}, float_precision="round_trip")
        sg_audit.analyse(inputs, output)
        if args.field_measurements:
            measurements = pd.read_csv(args.field_measurements, dtype={"case_id": str, "crop_id": str})
            if set(zip(measurements.crop_id, measurements.case_id)) != set(zip(inputs.crop_id, inputs.case_id)) or measurements.crop_id.duplicated().any():
                parser.error("Field measurement records must match the composition input fields")
            sg_audit.measurement_summary(measurements).to_csv(output / "measurement_summary.csv", index=False)
    print("Analysis completed. Outputs were written outside the code checkout.")


if __name__ == "__main__":
    main()
