from ..lib import np, signal, interpolate
from .transforms import calculate_improved_rms


def match_rms_ms(target_mid, target_side, reference_mid, reference_side, sample_rate, config):
    def match_rms(target, reference):
        t = calculate_improved_rms(target, sample_rate, config)
        r = calculate_improved_rms(reference, sample_rate, config)
        gain = r / max(t, config.epsilon)
        return target * gain

    target_side_rms = calculate_improved_rms(target_side, sample_rate, config)
    target_mid_rms = calculate_improved_rms(target_mid, sample_rate, config)
    mono_threshold = 0.01
    matched_mid = match_rms(target_mid, reference_mid)
    if target_side_rms / max(target_mid_rms, config.epsilon) < mono_threshold:
        mid_scale = np.max(np.abs(matched_mid)) / max(np.max(np.abs(target_mid)), config.epsilon)
        matched_side = target_side * mid_scale
    else:
        matched_side = match_rms(target_side, reference_side)
    return matched_mid, matched_side


def _mean_stft_mag(audio, sample_rate, fft_size):
    _, _, specs = signal.stft(
        audio,
        sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size // 2,
        boundary=None,
        padded=False,
    )
    return np.abs(specs).mean(axis=1)


def _smooth_ratio(ratio, config):
    n = len(ratio)
    grid_linear = np.linspace(0, config.internal_sample_rate / 2, n)
    grid_log = np.logspace(np.log10(20), np.log10(config.internal_sample_rate / 2), n)
    interp = interpolate.interp1d(grid_linear, ratio, kind="linear", bounds_error=False, fill_value="extrapolate")
    ratio_log = interp(grid_log)
    kernel = np.ones(9, dtype=np.float64) / 9.0
    smoothed = np.convolve(ratio_log, kernel, mode="same")
    interp_back = interpolate.interp1d(grid_log, smoothed, kind="linear", bounds_error=False, fill_value="extrapolate")
    out = interp_back(grid_linear)
    out[0] = 0
    if n > 1:
        out[1] = ratio[1]
    return out


def _fit_fir(target, reference, config, max_boost_db):
    target = np.maximum(target, config.min_value)
    ratio = reference / target
    lim = 10 ** (max_boost_db / 20.0)
    ratio = np.clip(ratio, 1.0 / lim, lim)
    ratio = _smooth_ratio(ratio, config)
    fir = np.fft.irfft(ratio)
    fir = np.fft.ifftshift(fir) * signal.windows.hann(len(fir))
    return fir


def match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config):
    sr = config.internal_sample_rate * config.oversampling_factor
    tm = _mean_stft_mag(target_mid, sr, config.fft_size)
    ts = _mean_stft_mag(target_side, sr, config.fft_size)
    rm = _mean_stft_mag(reference_mid, sr, config.fft_size)
    rs = _mean_stft_mag(reference_side, sr, config.fft_size)

    fir_m = _fit_fir(tm, rm, config, max_boost_db=3)
    fir_s = _fit_fir(ts, rs, config, max_boost_db=2)

    out_m = signal.oaconvolve(target_mid, fir_m, mode="same")
    out_s = signal.oaconvolve(target_side, fir_s, mode="same")

    freqs = np.linspace(0, sr / 2, len(out_m))
    mix = 0.98 * (1 - np.exp(-freqs / 10.0)) * np.exp(-freqs / 100000.0)
    out_m = (1 - mix) * target_mid + mix * out_m
    out_s = (1 - mix) * target_side + mix * out_s
    return out_m, out_s


def gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config):
    def apply(target, reference):
        target_rms = np.sqrt(np.mean(target**2) + config.epsilon)
        reference_rms = np.sqrt(np.mean(reference**2) + config.epsilon)
        gain = np.clip(reference_rms / target_rms, 0.5, 2.0)
        return target * gain ** (1 / config.rms_correction_steps)

    for _ in range(config.rms_correction_steps):
        target_mid = apply(target_mid, reference_mid)
        target_side = apply(target_side, reference_side)
    return target_mid, target_side

