from ..lib import np, signal, time
from .audio_io import load_audio, save_audio, apply_dither
from .transforms import (
    lr_to_ms,
    ms_to_lr,
    oversample,
    downsample,
    improved_anti_aliasing_filter,
    apply_lowpass_filter,
    add_subtle_mid_channel_saturation,
    calculate_improved_rms,
)
from .spectral import match_rms_ms, match_frequencies_ms, gradual_level_correction
from .stereo import finalize_stereo_image, analyze_stereo_width
from .dynamics import multi_stage_limiter
from ..cpu.reference import load_genre_profile, calculate_lufs, create_reference_from_profile, apply_eq_style, low_shelf_tighten


def calculate_true_peak(audio, sample_rate):
    upsampled = signal.resample_poly(audio, 4, 1, axis=-1)
    peak = np.max(np.abs(upsampled))
    return 20 * np.log10(max(peak, 1e-12))


def apply_true_peak_limit(audio, sample_rate, target_dbtp=-0.4, max_iters=2):
    result = audio
    for _ in range(max_iters):
        tp = calculate_true_peak(result, sample_rate)
        if tp <= target_dbtp:
            break
        result = result * (10 ** ((target_dbtp - tp) / 20.0))
    return result


def log_audio_metrics(audio, name, config):
    print(f"--- [cpu_test] {name} ---")
    print(f"shape={audio.shape}, max={np.max(np.abs(audio)):.4f}, lufs={calculate_lufs(audio, config.internal_sample_rate):.2f}")
    mid, side = lr_to_ms(audio)
    print(f"mid_rms={np.sqrt(np.mean(mid**2)):.4f}, side_rms={np.sqrt(np.mean(side**2)):.4f}, width={analyze_stereo_width(mid, side):.4f}")


def process_audio(target, reference, config, genre_profile=None):
    t0 = time.time()
    initial_loudness_ratio = 1.0
    if config.loudness_option == "dynamic":
        initial_loudness_ratio = 0.80
    elif config.loudness_option == "soft":
        initial_loudness_ratio = 0.70
    elif config.loudness_option == "loud":
        initial_loudness_ratio = 1.20
    target = target * initial_loudness_ratio

    target = target.astype(np.float64)
    reference = reference.astype(np.float64)

    target = oversample(target, config.oversampling_factor)
    reference = oversample(reference, config.oversampling_factor)
    oversampled_rate = config.internal_sample_rate * config.oversampling_factor

    target = improved_anti_aliasing_filter(target, oversampled_rate)
    reference = improved_anti_aliasing_filter(reference, oversampled_rate)

    target_mid, target_side = lr_to_ms(target)
    reference_mid, reference_side = lr_to_ms(reference)

    target_mid, target_side = match_rms_ms(target_mid, target_side, reference_mid, reference_side, oversampled_rate, config)
    processed_mid_side = ms_to_lr(target_mid, target_side)
    processed_rms = calculate_improved_rms(processed_mid_side, oversampled_rate, config)
    ref_rms = calculate_improved_rms(reference, oversampled_rate, config)
    print(f"[cpu_test] RMS matched: processed={processed_rms:.6f}, reference={ref_rms:.6f}")

    target_mid = add_subtle_mid_channel_saturation(target_mid, config)
    target_mid, target_side = match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config)
    target_side = low_shelf_tighten(target_side, config.internal_sample_rate, cutoff_freq=100, gain=0.5, order=4)
    if config.eq_style != "Neutral":
        target_mid, target_side = apply_eq_style(target_mid, target_side, config.internal_sample_rate, config.eq_style)

    result = ms_to_lr(target_mid, target_side)
    result = apply_lowpass_filter(result, config)
    target_mid, target_side = lr_to_ms(result)
    target_mid, target_side = gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config)
    result = finalize_stereo_image(target_mid, target_side, reference_mid, reference_side, config)

    result *= initial_loudness_ratio
    result = multi_stage_limiter(result, config)
    # Keep original-safe headroom before downsampling.
    result = np.clip(result, -0.95, 0.95)
    result = downsample(result, config.oversampling_factor, oversampled_rate)

    target_peak_db = -0.5
    current_peak_db = 20 * np.log10(max(np.max(np.abs(result)), 1e-12))
    if current_peak_db < target_peak_db:
        result *= 10 ** ((target_peak_db - current_peak_db) / 20.0)

    # Apply true-peak safety before dithering.
    result = apply_true_peak_limit(result, config.internal_sample_rate, target_dbtp=-0.4, max_iters=2)

    if genre_profile and genre_profile["genre"] in ["Piano", "Orchestral", "Speech"]:
        if config.loudness_option == "dynamic":
            result *= 10 ** (-0.3 / 20)
        elif config.loudness_option == "soft":
            result *= 10 ** (-0.6 / 20)

    # Dither at the very end.
    result = apply_dither(result)
    # Final PCM safety clip only; should rarely engage after TP limiting.
    result = np.clip(result, -0.999, 0.999)

    print(f"[cpu_test] process_audio time={time.time()-t0:.2f}s, true_peak={calculate_true_peak(result, config.internal_sample_rate):.2f} dBTP")
    return result


def master_audio(input_file, output_file, config, eq_style, is_preview=False):
    t0 = time.time()
    config.eq_style = eq_style
    target, sr = load_audio(input_file, config)
    print(f"[cpu_test] input duration={len(target[0]) / sr:.2f}s")

    if is_preview:
        preview_samples = min(sr * 30, target.shape[1])
        target = target[:, :preview_samples]

    if config.genre:
        genre_profile = load_genre_profile(config.genre)
        reference, _ = create_reference_from_profile(genre_profile, config)
    elif config.reference_file:
        reference, _ = load_audio(config.reference_file, config)
        genre_profile = None
    else:
        raise ValueError("Either genre or reference file must be specified")

    log_audio_metrics(reference, "reference", config)
    processed = process_audio(target, reference, config, genre_profile)
    save_audio(processed, output_file, sr)
    print(f"[cpu_test] Total Python processing time: {time.time()-t0:.2f} seconds")
