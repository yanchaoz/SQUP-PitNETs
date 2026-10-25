# Analysis conventions

## Density sampling

Virtual 20 x 20 micrometre fields pool adjacent 10 x 10 micrometre cells.
Candidates require at least 50% valid tissue area. Four lattice offsets define
non-overlapping candidate groups. For each draw, an eligible group is selected
uniformly and distinct fields are sampled without replacement. Field counts are
1, 3, 5, 10, 20, 50 and 100 where feasible; repeats default to 1,000.

The estimate is pooled object count divided by pooled actual valid area, not the
mean of field densities. Absolute relative error is
`100 * abs(sampled_density - full_density) / full_density`. Errors are first
summarised by the median within each case; cohort median/IQR are then calculated
over case medians. All supplied specimens contribute when feasible.

## Composition sampling

Candidate sides are 20 and 50 micrometres; whole fields must fit inside the
physical domain and contain at least 50% valid tissue. Lattice offsets use the
10-micrometre base grid. The smaller field uses one field per draw; the larger
uses 1, 2, 4, 8, 16, 32 and 64 where feasible.

Only C1-C4 count composition enters this experiment, independently normalised
after excluding C5. TVD is `0.5 * sum(abs(sample_fraction - full_fraction))`.
Coverage is the first tested nominal area with TVD <= 0.05 in at least 90% of
draws, sustained at all larger feasible tested areas. No interpolation is used.

Enrichment means an unrounded count fraction strictly greater than the external
reference fraction. Among enriched states, the second-largest full-region count
fraction defines the secondary component. Under-call means its sample fraction
is no longer greater than the reference fraction; absence means zero sampled
profiles. Empty samples have undefined TVD and count as failures for the
stability, under-call and absence calculations.

The externally identified reference is excluded from composition cohort
summaries. Its full linked counts define the comparison fractions. Available
case counts are derived separately for each feasible field count.

## Random numbers

The generator uses NumPy default_rng. Seeds are derived from the supplied seed
and condition identifiers using the first eight bytes of a BLAKE2b digest in
little-endian order, modulo 2**32 - 1. The density generator advances across
successive field counts within each case/organelle. Composition uses a separate
generator for each condition. Changing identifiers, candidate pools, input area
versions or enumeration order can change the realised draws.

## SG review-field agreement

Count- and full-profile-area-weighted C1-C4 fractions are calculated separately
on both sides. Unmatched profiles remain included. Signed percentage-point
changes, absolute changes and TVD are calculated per field before summarisation;
opposite field changes are not pooled so that they cancel. A secondary summary
uses medians within each case followed by a summary across those case medians.

Optional measurement summaries take the absolute supplied count and occupied-
area error percentages. This is internal measurement agreement when inputs share
production boundaries, not independent instance-level accuracy. This summary
command does not relabel objects, fit a new atlas or select fields based on
favourable errors. The separate core commands are described in [CORE.md](CORE.md).

The atlas feature names are area, perimeter, maximum principal-axis span,
major-axis length, aspect ratio, extent, convexity and axis skewness. The legacy
column `feret_max_px` refers to maximum principal-axis span, not an all-angle
Feret diameter. `core.py features` implements these eight archived definitions;
the SG summary command consumes already computed aggregates.

## Exact area kernel

`exact_grid(layout, masks, ye, xe, bounds)` accepts binary tissue masks (zero
denotes valid tissue), tile physical bounds, grid edges and the analysis-domain
bounds. Mask pixels are 0.08 micrometres. Intersections use integer nanometres;
areas are returned in square micrometres. Partial tile/grid pixels are weighted
by physical overlap, not independently rounded outward. The implementation
checks non-overlapping tile coverage, valid-area bounds and conservation.

Mask values, calibration and coordinates must be verified upstream. Synthetic
area tests are software checks only, not validation of tissue segmentation.
