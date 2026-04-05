from ..lib import np
from .transforms import lr_to_ms, ms_to_lr


def analyze_stereo_width(mid, side):
    mid_energy = np.mean(np.square(mid))
    side_energy = np.mean(np.square(side))
    return side_energy / (mid_energy + side_energy + 1e-8)


def finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config):
    initial_width = analyze_stereo_width(target_mid, target_side)
    reference_width = analyze_stereo_width(reference_mid, reference_side)
    if initial_width < 0.02:
        return ms_to_lr(target_mid, target_side)
    width_diff = reference_width - initial_width
    max_adjustment = 0.15
    if width_diff > 0:
        factor = 1 + min(width_diff / max(initial_width, 1e-8), max_adjustment)
        adjusted_side = target_side * factor
    else:
        adjusted_side = target_side
    result = ms_to_lr(target_mid, adjusted_side)
    initial_rms = np.sqrt(np.mean(target_mid**2 + target_side**2))
    current_rms = np.sqrt(np.mean(result**2))
    result *= initial_rms / max(current_rms, 1e-8)
    return result

