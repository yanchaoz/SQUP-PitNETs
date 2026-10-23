# External input schemas

No populated inputs or study identifiers are included. All file locations and
the reference identifier are supplied by the operator. CSV.GZ denotes ordinary
gzip-compressed CSV. Identifiers are strings, not necessarily numeric.

## Spatial inputs

Place the following files in the external `--input-dir`:

- `case_order.json`: nonempty ordered list of unique specimen identifiers.
  Identifiers may contain ASCII letters, digits, underscores and hyphens. The
  order is part of the reproducibility configuration.
- `case_domains.csv`: one row per `case_id,organelle`; `organelle` is `SG` or
  `Mito`. Required fields are `n_profiles`, `valid_area_um2`,
  `full_density_per_um2`, `height_um`, `width_um`. SG and Mito can have distinct
  domains. Density is count divided by that branch's actual valid tissue area.
- `density_grid.csv.gz`: one row per `case_id,organelle,grid_i,grid_j`, with
  `count`, `valid_area_um2`, `y0_um,y1_um,x0_um,x1_um`, `candidate_cell`.
  Indices are zero-based and refer to the nominal 10-micrometre base grid.
  `candidate_cell` is a binary flag for the density-sampling candidate domain;
  rows with zero flags still contribute to the complete reference totals.
- `composition_grid.csv.gz`: one row per `case_id,grid_i,grid_j`, with integer
  `C1,C2,C3,C4` counts and `valid_area_um2`. C5 and objects without accepted
  coordinate linkage are not imputed. The complete linked inventory, not an
  unlinked full inventory, defines the spatial composition reference.

Counts must be finite nonnegative integers; areas must be finite nonnegative
values in square micrometres. Domain totals must agree with density-grid totals.
In corrected mode, valid area cannot exceed the corresponding physical cell
support. No specimen counts or observation totals are hard-coded.

## SG compact composition input

One row for every `crop_id,case_id,side,display_class` combination. Each field
requires both `reference` and `prediction` sides and classes 1-5, including rows
with zero counts. Required measurement columns:

- `n_profiles`
- `full_profile_area_px`
- `matched_both_included_n`
- `unmatched_n`
- `counterpart_outside_inclusion_n`

The three status counts must sum to `n_profiles` in every row. Both-included
matched totals must agree between sides within each field. Areas are full areas
of eligible centroid-owned profiles, not foreground pixels clipped to the core.
Upstream contour eligibility and matching must be established before supplying
these aggregates; this package does not independently verify those operations.

## Optional SG field-measurement input

One row per `crop_id,case_id`, covering exactly the same fields as the compact
composition input, with `signed_count_bias_pct` and
`signed_occupied_area_error_pct`. These must already be computed as signed
percentage errors relative to each field's reference measurement. The command
summarises their absolute values; it does not redo watershed or instance matching.

## Outputs

Spatial outputs include per-case sampling CSVs, across-case density summaries,
required-coverage CSVs, linked full-region inventories and per-resample NPZs.
The JSON summary contains values calculated from the supplied inputs. SG outputs
include per-field and case-median composition differences and summary CSVs.

Outputs can contain study identifiers and observations. Keep them private unless
there is separate authorisation to release them. Input and output directories
must be separate and must not be nested within each other.
