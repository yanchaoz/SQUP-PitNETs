# SQUP-PitNETs

Code-only release of downstream analysis methods for a PitNET large-field SEM
study. Research data are supplied separately and are **not included** here.

## Scope

- Fixed-physical-area density and morphology-composition sampling.
- Count-weighted C1-C4 total variation distance, coverage requirements and
  secondary-component under-call calculations.
- SG review-field count/area composition comparisons, including unmatched
  profiles, plus optional summaries of supplied field-level measurement errors.
- Exact tissue-mask/grid intersection areas with fractional boundary handling.

This repository does not provide segmentation training, inference, watershed,
independent instance validation, atlas refitting or membrane-distance extraction.
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
```

GitHub Actions uses only synthetic tests. No research data or secrets are needed.

## Run with separately supplied data

Keep inputs and outputs outside this checkout. Replace the placeholders with
your own input location, output location and reference identifier:

```sh
python run.py spatial --input-dir /path/to/private/inputs --reference-id REFERENCE_ID --area-mode corrected --output-dir /path/to/private/results
python run.py sg --input-csv /path/to/private/composition_inputs.csv --output-dir /path/to/private/sg_results
```

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
