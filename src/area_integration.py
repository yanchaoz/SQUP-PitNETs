"""Fractional tissue-mask/grid intersections; mask value 0 is analysable tissue.

Geometry is clipped to tile bounds and the organelle-specific acquisition domain.
Original binary masks are not included in this grid-level reproduction archive.
"""
import numpy as np
from scipy.sparse import csr_matrix
SCALE=.08

def axis_weights(edges, origin, n, scale, upper):
    coordinates = np.r_[edges, origin, scale, upper]*1000
    assert np.max(abs(coordinates-np.rint(coordinates))) < 1e-6, 'Geometry not representable at 1 nm resolution'
    edges = np.rint(edges*1000).astype(np.int64)
    origin, scale, upper = [int(round(v*1000)) for v in (origin, scale, upper)]
    lo = origin + np.arange(n) * scale
    hi = np.minimum(lo + scale, upper)
    overlap = np.maximum(0, np.minimum(edges[1:, None], hi[None, :]) -
                         np.maximum(edges[:-1, None], lo[None, :]))
    return csr_matrix(overlap)

def exact_grid(layout, masks, ye, xe, bounds):
    result = np.zeros((len(ye)-1, len(xe)-1))
    covered = np.zeros_like(result)
    direct_area = 0.
    for r in layout.itertuples(index=False):
        if r.tile_y0_um >= bounds[0] or r.tile_x0_um >= bounds[1]:
            continue
        mask = masks[r.tile]
        inside = (mask == 0).astype(float)
        uy = min(r.tile_y1_um, bounds[0])
        ux = min(r.tile_x1_um, bounds[1])
        wy = axis_weights(ye, r.tile_y0_um, inside.shape[0], SCALE, uy)
        wx = axis_weights(xe, r.tile_x0_um, inside.shape[1], SCALE, ux)
        result += (wx @ (wy @ inside).T).T
        y0, x0, yn, xn = [int(round(v*1000)) for v in (r.tile_y0_um, r.tile_x0_um, uy, ux)]
        py = np.maximum(0, np.minimum(y0 + (np.arange(mask.shape[0])+1)*80, yn) -
                        np.maximum(y0 + np.arange(mask.shape[0])*80, 0))
        px = np.maximum(0, np.minimum(x0 + (np.arange(mask.shape[1])+1)*80, xn) -
                        np.maximum(x0 + np.arange(mask.shape[1])*80, 0))
        direct_area += float(py @ inside @ px)
        cy = np.maximum(0, np.minimum(ye[1:], uy) - np.maximum(ye[:-1], r.tile_y0_um))
        cx = np.maximum(0, np.minimum(xe[1:], ux) - np.maximum(xe[:-1], r.tile_x0_um))
        covered += cy[:, None]*cx[None, :]
    result /= 1e6
    direct_area /= 1e6
    cell_area = np.outer(np.diff(ye), np.diff(xe))
    assert np.max(covered-cell_area) < 1e-7, 'Acquisition tiles overlap physically'
    assert np.max(result-covered) < 1e-7, 'Valid mask area exceeds imaged coverage'
    assert abs(result.sum()-direct_area) < 1e-5, 'Area conservation failed'
    return result, covered, direct_area
