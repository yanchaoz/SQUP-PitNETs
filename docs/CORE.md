# Core algorithms

These modules are a small, portable extraction of the archived computation, not
a replacement neural-network implementation. All examples/tests are synthetic.
They do not measure segmentation accuracy. Only code and method parameters are
included: no image, contour, feature observation, fitted centre, covariance,
class assignment, patient identifier or study outcome is distributed.

## Instance reconstruction

`src/reconstruction.py` accepts aligned 2-D binary foreground and boundary masks
(0/1 or 0/255). Continuous probabilities are rejected; convert them using the
segmentation model's actual threshold before using this binary production route.
The CLI reads NPY, TIFF and singleton-3-D NIfTI, retaining in-plane array order.
It does not resample, reorient, mosaic or register images; verify calibration,
orientation and registration upstream. Output `instances.npy` uses pixel
coordinates; NIfTI affine/header metadata are not carried into NPY.

Defaults reproduce the archived binary route: foreground opening and closing
with disk radius 1, then hole filling. Seed components are foreground minus
boundary, labelled with 8-connectivity; seed components smaller than 16 pixels
are removed. Marker-controlled watershed propagates with 4-connectivity on the
negative binary foreground, within cleaned foreground. Instances smaller than
64 pixels are removed. No distance-transform elevation, contour-based expansion,
cross-tile stitching or instance-specific closing is added. A boundary pixel
may belong to a final instance: boundaries restrict seeds, not final support.
Pixels in foreground components with no surviving seed remain unassigned.
Instance IDs may have gaps; count unique positive IDs, never the maximum ID.
Edge-touching objects remain in reconstruction, but are excluded from morphology.

The archived boundary-aware watershed approach follows the MitoEM/connectomics
postprocessing family. This package expresses that restricted binary operation
using NumPy, SciPy and scikit-image calls; it does not vendor the connectomics,
nnUNet, SAM or DINO source trees. Dependencies retain their respective licences.

## Eight-feature extraction

`src/features.py` processes an instance label map. One positive ID denotes one
object. Background is zero. Raw intensity images are not required. Exactly eight
geometry descriptors are used, in this persisted order:

| Column | Definition |
| --- | --- |
| `aspect_ratio` | Regionprops equivalent-ellipse major/minor axis length. |
| `extent` | Object pixel area divided by its axis-aligned bounding-box area. |
| `convexity` | Number of convex-hull boundary pixels after disk-1 erosion, divided by regionprops perimeter. This archived proxy is not area/convex-hull area and is not clipped to 1. |
| `axis_skewness` | SciPy biased skewness of positive transverse widths in bins along the principal axis. This is not coordinate skewness. |
| `area_px` | Pixel count (units: square pixels). |
| `perimeter_px` | Regionprops pixel-centre contour perimeter estimate (units: pixels). |
| `feret_max_px` | Pixel-centre projection span along the first principal axis, not an all-angle maximum Feret diameter. The legacy field name is retained for compatibility. |
| `major_axis_px` | Equivalent-ellipse major-axis length (units: pixels). |

Principal axes come from the eigenvectors of the pixel-coordinate covariance.
The transverse-width calculation uses `max(10, ceil(principal_span))` equal
bins; each width is the range of minor-axis coordinates in that bin. Only
positive widths enter skewness, and at least eight such bins are required.
The hull calculation uses the archived 8-pixel padded bounding box, clipped to
image bounds. Location metadata use zero-based `(row, column)` coordinates and
half-open bounding boxes. Metadata are not additional clustering features.

`core.py features` emits accepted `features.csv` and `excluded_features.csv`
with explicit reasons. It runs eligibility on ONE tile: area >=50 pixels,
no image-border contact, all eight features finite, and optional MAD filtering.
Per feature, robust z = `0.6745 * (value - median) / MAD`; a dimension is flagged
when absolute z >10. A profile is removed only if at least seven dimensions are
flagged. A dimension with fewer than five finite values or MAD <1e-12 is not
flagged. Apply this before pooling tiles, not on the pooled reference table.
The extraction function can expose border objects for inspection, but they must
not silently enter atlas fitting. Pure count/density inventories are separate
from this morphology-eligible subset.

Use consistent pixel size across datasets. Do not supply micrometre-converted
features to a scaler fitted in pixels. This release does not claim rotational
or resolution invariance for the archived pixel-grid descriptors.

## Reference fitting and prototypes

`src/atlas.py` consumes an externally supplied eligible reference feature table.
It fits median imputation and a StandardScaler, then repeated K-means in the
standardised eight-dimensional space. UMAP does not participate. Defaults:
six clusters, 20 runs, 20 initialisations per run, at most 300 iterations, seed42.
Run i uses seed `seed + 17*i + 131*n_clusters`. The minimum-inertia run is the
alignment reference. Hungarian matching minimises Euclidean centre distances;
aligned labels vote per profile. The lowest label wins a vote tie. If consensus
loses a cluster, the archived rule falls back to reference-run labels with unit
confidence; `consensus_fallback` records that event. Such unit confidence is not
evidence of perfect cross-run stability. Labels are subsequently ranked by
decreasing cluster population and remain arbitrary numeric labels.

Each source cluster is partitioned by K-means into up to three prototypes,
using seed `seed + 97*source_label`, 20 initialisations and 300 iterations.
The number is reduced for small clusters using a minimum of 10 members per
requested prototype. Each centre is the arithmetic mean of its assigned
members, not a real exemplar. Covariance precision uses Ledoit-Wolf for >=8
members; smaller/failing groups use diagonal inverse sample variance, replacing
variance <1e-6 or nonfinite variance with 1. A singleton uses identity precision.

The safe NPZ representation contains `features`, `median`, `mean`, `scale`,
`centers`, `precisions`, `labels` and optionally `config` (a JSON string).
`centers` has shape P x 8, `precisions` P x 8 x 8, `labels` P. Preprocessing
arrays each have length 8. All fitted arrays are research-derived and must be
supplied separately. No pickle is loaded. Refit labels need external inspection
and mapping; a newly fitted label number is not a fixed morphology/hormone name.

## Frozen transfer

`src/assignment.py` uses the frozen imputer/scaler, not application-set fitting.
For prototype p it calculates
`d_p = sqrt((x-centre_p)^T precision_p (x-centre_p))`.
The nearest five prototypes (or all if fewer exist) contribute weights
`1/(d_p + 1e-8)`. Weights are summed by source label; the maximum vote wins.
`prototype_vote_share` is the winning weight divided by total selected weight.
It is a relative support statistic, not a calibrated probability. Voting ties
follow the first class encountered in distance order. The retained prototype ID
and distance refer to the closest selected prototype OF THE WINNING CLASS,
which need not be the closest individual prototype overall.

This minimal implementation provides the final no-rejection route only, not the
historical distance/vote threshold variants. The optional `--class-map` is an
external JSON object mapping source-label strings to integer display labels.
Mapping occurs AFTER source voting, never before voting. No named biological
mapping or clinical decision rule is inferred by these algorithms.

## Reproducibility boundary

Defaults above are algorithm parameters recovered from archived computation,
not new performance claims. Exact study replay needs its external eligible
feature rows, frozen model, mapping, original input order and software versions.
The dependency pins specify the tested release environment, not a retrospective
claim about every production run. Reference fitting does not replace a frozen
atlas when checking old application outputs. Downstream grid creation,
coordinate linkage and membrane distance remain external steps. The four core
modules do not themselves constitute independent biological validation.
