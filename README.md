# SQUP-PitNETs

Code-only release of core reconstruction/geometry/atlas algorithms and downstream
analysis methods. Research data and fitted models are supplied separately and
are **not included** here.

## Scope

- Boundary-aware 2-D instance reconstruction from binary semantic predictions.
- Eight morphology features, object locations and tile-wise eligibility filters.
- Standardised-feature consensus K-means and within-cluster prototype centres.
- Frozen local-Mahalanobis nearest-prototype voting with external class mappings.
- Fixed-physical-area density and morphology-composition sampling.
- Count-weighted C1-C4 total variation distance, coverage requirements and
  secondary-component under-call calculations.
- SG review-field count/area composition comparisons, including unmatched
  profiles, plus optional summaries of supplied field-level measurement errors.
- Exact tissue-mask/grid intersection areas with fractional boundary handling.

This repository does not provide neural-network training/inference, independent
instance validation, membrane-distance extraction or study-fitted model values.
It does not contain case identifiers, observed measurements, clinical records,
images, checkpoints, manuscript templates, archived result tables or private
provenance files. Generated test fixtures are explicitly synthetic software
examples, not experimental validation data.

## Install and test

Python 3.10 or 3.11:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --help
python core.py --help
```

GitHub Actions uses only synthetic tests. No research data or secrets are needed.

## Run with separately supplied data

Keep inputs and outputs outside this checkout. Replace the placeholders with
your own input location, output location and reference identifier:

```sh
python core.py instances --foreground /path/to/inputs/foreground.tif --boundary /path/to/inputs/boundary.tif --output-dir /path/to/instances
python core.py features --labels /path/to/instances/instances.npy --output-dir /path/to/features
python core.py fit --input-csv /path/to/inputs/reference_features.csv --output-dir /path/to/model
python core.py assign --input-csv /path/to/inputs/application_features.csv --atlas /path/to/model/atlas.npz --output-dir /path/to/assigned
python run.py spatial --input-dir /path/to/private/inputs --reference-id REFERENCE_ID --area-mode corrected --output-dir /path/to/private/results
python run.py sg --input-csv /path/to/private/composition_inputs.csv --output-dir /path/to/private/sg_results
```

The first two core commands operate on one registered 2-D field at a time.
Concatenate eligible feature rows with unique case/tile/object keys before
reference fitting or application transfer. Use the same pixel calibration and
feature order for reference and application profiles. Existing study assignments
must use the separately supplied **frozen** atlas, not a new refit. Cluster labels
from a refit are arbitrary; no hormone names or study-specific class mapping are
embedded. See [core algorithm details](docs/CORE.md) for conventions and schemas.

The SG command optionally accepts `--field-measurements` for a separate per-field
measurement CSV. The input schemas are described in [docs/INPUTS.md](docs/INPUTS.md).
There is no data download endpoint or automatic retrieval in this code release.
Actual research data must be supplied separately by the authors.

## Numerical conventions

`corrected` is the default area mode. It requires corrected valid-tissue grid
areas and uses integer square-nanometre arithmetic for eligibility thresholds.
It does not repair incorrect input areas automatically. The exact mask-area
kernel is available separately in `src/area_integration.py`.

`legacy` retains historical floating-point threshold arithmetic solely for
replaying separately supplied archived inputs. It is not a recommendation to
reintroduce grid-boundary overcounting. Do not mix densities and denominators
from different area versions.

The reference identifier, case order and observations are external inputs.
Sampling repeats and seed values are CLI parameters with method defaults. Density
curves include all supplied specimens; composition cohort summaries exclude the
supplied reference. Technical resamples are nested within cases, not independent
biological replicates. See [docs/METHODS.md](docs/METHODS.md).

## Data handling

The CLI refuses to write analysis outputs inside this checkout. `.gitignore`
also excludes common research input/output formats. These are safeguards, not
a replacement for reviewing files before publishing.

Only code, dependency specifications, documentation and synthetic test logic are
released. No new software or data licence is assigned by this PR.
