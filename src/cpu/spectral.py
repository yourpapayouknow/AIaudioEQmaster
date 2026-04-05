from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt
from .transforms import calculate_improved_rms

def match_rms_ms(target_mid, target_side, reference_mid, reference_side, sample_rate, config):
    def match_rms(target, reference):
        target_rms = calculate_improved_rms(target, sample_rate, config)
        reference_rms = calculate_improved_rms(reference, sample_rate, config)
        gain = reference_rms / target_rms
        return target * gain

    # Calculate the RMS of the target side channel
    target_side_rms = calculate_improved_rms(target_side, sample_rate, config)
    target_mid_rms = calculate_improved_rms(target_mid, sample_rate, config)

    # Define a threshold for considering the audio as "nearly mono"
    mono_threshold = 0.01  # Adjust this value as needed

    matched_mid = match_rms(target_mid, reference_mid)

    if target_side_rms / target_mid_rms < mono_threshold:
        print("Detected nearly mono audio. Skipping side channel RMS matching.")
        # Scale the side channel by the same factor as the mid channel
        mid_scale_factor = np.max(np.abs(matched_mid)) / np.max(np.abs(target_mid))
        matched_side = target_side * mid_scale_factor
    else:
        matched_side = match_rms(target_side, reference_side)

    return matched_mid, matched_side

def match_frequencies_ms(target_mid, target_side, reference_mid, reference_side, config):
    def calculate_average_fft(*args, sample_rate, fft_size, config):
        if len(args) == 1:
            # Single audio input
            audio = args[0]
            mid = side = audio
        elif len(args) == 2:
            # Separate mid and side inputs
            mid, side = args
        else:
            raise ValueError("Invalid number of arguments for calculate_average_fft")

        if config.use_loudest_parts:
            segment_length = sample_rate // 10  # 100ms segments
            num_segments = len(mid) // segment_length
            segments_mid = np.array_split(mid[:num_segments * segment_length], num_segments)
            
            # Calculate RMS based on mid channel only
            segment_rms = np.sqrt(np.mean(np.square(segments_mid), axis=1))
            
            loud_mask = segment_rms > (config.loudness_threshold * np.max(segment_rms))
            loud_segments_mid = [seg for seg, is_loud in zip(segments_mid, loud_mask) if is_loud]
            
            if len(loud_segments_mid) == 0:
                print("No segments above threshold, using entire audio.")
                return mid, side
            else:
                percentage_used = (len(loud_segments_mid) / len(segments_mid)) * 100
                print(f"Using {percentage_used:.2f}% of the audio (threshold: {config.loudness_threshold})")
            
            # Use the same mask for side channel
            segments_side = np.array_split(side[:num_segments * segment_length], num_segments)
            loud_segments_side = [seg for seg, is_loud in zip(segments_side, loud_mask) if is_loud]
            
            mid = np.concatenate(loud_segments_mid)
            side = np.concatenate(loud_segments_side)

        _, _, specs_mid = signal.stft(
            mid,
            sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size // 2,
            boundary=None,
            padded=False,
        )
        _, _, specs_side = signal.stft(
            side,
            sample_rate,
            window="hann",
            nperseg=fft_size,
            noverlap=fft_size // 2,
            boundary=None,
            padded=False,
        )
        return np.abs(specs_mid).mean(axis=1), np.abs(specs_side).mean(axis=1)

    def smooth_spectrum(spectrum, config):
        fft_size = (len(spectrum) - 1) * 2
        grid_linear = np.linspace(0, config.internal_sample_rate / 2, len(spectrum))
        grid_logarithmic = np.logspace(
            np.log10(4 * config.internal_sample_rate / fft_size),
            np.log10(config.internal_sample_rate / 2),
            (len(spectrum) - 1) * config.lin_log_oversampling + 1,
        )

        interpolator = interpolate.interp1d(grid_linear, spectrum, "cubic", bounds_error=False, fill_value="extrapolate")
        spectrum_log = interpolator(grid_logarithmic)

        spectrum_smoothed = lowess(
            spectrum_log,
            np.arange(len(spectrum_log)),
            frac=config.lowess_frac,
            it=config.lowess_it,
            delta=config.lowess_delta * len(spectrum_log),
        )[:, 1]

        interpolator = interpolate.interp1d(
            grid_logarithmic, spectrum_smoothed, "cubic", bounds_error=False, fill_value="extrapolate"
        )
        spectrum_filtered = interpolator(grid_linear)

        spectrum_filtered[0] = 0
        spectrum_filtered[1] = spectrum[1]

        return spectrum_filtered

    def get_fir(target_mid, target_side, reference_mid, reference_side, config, is_side=False):
        target_fft_mid, target_fft_side = calculate_average_fft(
            target_mid, target_side, 
            sample_rate=config.internal_sample_rate * config.oversampling_factor, 
            fft_size=config.fft_size, 
            config=config
        )
        reference_fft_mid, reference_fft_side = calculate_average_fft(
            reference_mid, reference_side, 
            sample_rate=config.internal_sample_rate * config.oversampling_factor, 
            fft_size=config.fft_size, 
            config=config
        )
        
        target_fft = target_fft_side if is_side else target_fft_mid
        reference_fft = reference_fft_side if is_side else reference_fft_mid
        
        target_fft = np.maximum(target_fft, config.min_value)
        matching_fft = reference_fft / target_fft
        
        max_boost_db = 2 if is_side else 4  # Further reduced max boost
        matching_fft = np.clip(matching_fft, 10**(-max_boost_db/20), 10**(max_boost_db/20))
        
        matching_fft_filtered = smooth_spectrum(matching_fft, config)
        
        # Apply softer bass preservation
        bass_freq = config.bass_preservation_freq
        bass_blend = config.bass_preservation_blend
        
        # Create a gentler transition curve
        freqs = np.linspace(0, config.internal_sample_rate/2, len(matching_fft))
        bass_preservation = 1 - (1 - bass_blend) * (1 / (1 + (freqs / bass_freq)**2))
        
        # Apply the softer bass preservation
        matching_fft = 1 + (matching_fft - 1) * bass_preservation
        
        fir = np.fft.irfft(matching_fft_filtered)
        fir = np.fft.ifftshift(fir) * signal.windows.hann(len(fir))
        
        return fir

    mid_fir = get_fir(target_mid, target_side, reference_mid, reference_side, config, is_side=False)
    side_fir = get_fir(target_mid, target_side, reference_mid, reference_side, config, is_side=True)

    result_mid = signal.fftconvolve(target_mid, mid_fir, mode="same")
    result_side = signal.fftconvolve(target_side, side_fir, mode="same")

    def frequency_dependent_mix(freq, low_freq=10, high_freq=100000):
        return 0.99 * (1 - np.exp(-freq/low_freq)) * np.exp(-freq/high_freq)

    freqs = np.linspace(0, config.internal_sample_rate * config.oversampling_factor / 2, len(result_mid))
    mix = frequency_dependent_mix(freqs)
    result_mid = (1 - mix) * target_mid + mix * result_mid
    result_side = (1 - mix) * target_side + mix * result_side

    return result_mid, result_side

def gradual_level_correction(target_mid, target_side, reference_mid, reference_side, config):
    def apply_correction(target, reference):
        target_rms = np.sqrt(np.mean(target**2) + config.epsilon)
        reference_rms = np.sqrt(np.mean(reference**2) + config.epsilon)
        gain = np.clip(reference_rms / target_rms, 0.5, 2.0)
        return target * gain ** (1 / config.rms_correction_steps)

    for step in range(config.rms_correction_steps):
        target_mid = apply_correction(target_mid, reference_mid)
        target_side = apply_correction(target_side, reference_side)

    return target_mid, target_side
