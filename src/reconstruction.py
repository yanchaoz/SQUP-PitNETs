"""2-D instance reconstruction from already binarised foreground/boundary masks."""
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.segmentation import watershed


def binary_mask(array):
    array = np.asarray(array)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("Expected a nonempty 2-D mask")
    if not np.isin(array, [0, 1, 255]).all():
        raise ValueError("Supply a binary mask (0/1 or 0/255), not probabilities")
    return array > 0


def remove_small_labels(labels, minimum):
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[(ids != 0) & (counts < minimum)]
    result = labels.copy()
    result[np.isin(labels, small)] = 0
    return result


def reconstruct(foreground, boundary, opening_radius=1, closing_radius=1,
                fill_holes=True, min_seed_size=16, min_instance_size=64):
    """Return instance IDs; edge objects remain, and IDs need not be contiguous.

    The archived production route binarises both semantic outputs before
    watershed. Boundary pixels exclude seeds, not the final foreground support.
    Seed components use 8-connectivity; watershed propagation uses 4-connectivity.
    """
    foreground, boundary = binary_mask(foreground), binary_mask(boundary)
    if foreground.shape != boundary.shape:
        raise ValueError("Foreground and boundary must share the same pixel grid")
    for value in (opening_radius, closing_radius, min_seed_size, min_instance_size):
        if int(value) != value or value < 0:
            raise ValueError("Radii and size thresholds must be nonnegative integers")
    if opening_radius:
        foreground = binary_opening(foreground, disk(opening_radius))
    if closing_radius:
        foreground = binary_closing(foreground, disk(closing_radius))
    if fill_holes:
        foreground = binary_fill_holes(foreground)
    markers = remove_small_labels(label(foreground & ~boundary), min_seed_size)
    # A flat binary foreground is intentional; do not replace it with an EDT.
    instances = watershed(-255.0 * foreground, markers, mask=foreground)
    return remove_small_labels(instances, min_instance_size).astype(np.int32)
