from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt

def lr_to_ms(array):
    mid = (array[0] + array[1]) * 0.5
    side = (array[0] - array[1]) * 0.5
    return mid, side

def ms_to_lr(mid, side):
    min_length = min(len(mid), len(side))
    mid = mid[:min_length]
    side = side[:min_length]
    
    left = mid + side
    right = mid - side
    return np.vstack((left, right))

def oversample(audio, factor):
    return signal.resample_poly(audio, factor, 1, axis=-1)

def downsample(audio, factor, sample_rate):
    filtered_audio = improved_anti_aliasing_filter(audio, sample_rate)
    return signal.resample_poly(filtered_audio, 1, factor, axis=-1)

def improved_anti_aliasing_filter(audio, sample_rate):
    nyquist = sample_rate / 2
    cutoff = 0.9 * nyquist
    sos = signal.butter(10, cutoff / nyquist, btype='low', output='sos')
    
    filtered_audio = np.zeros_like(audio)
    for channel in range(audio.shape[0]):
        filtered_audio[channel] = signal.sosfilt(sos, audio[channel])
    
    return filtered_audio

def apply_lowpass_filter(audio, config):
    nyquist = config.internal_sample_rate * config.oversampling_factor / 2
    sos = signal.butter(10, config.lowpass_cutoff / nyquist, btype='low', output='sos')
    
    filtered_audio = np.zeros_like(audio)
    for channel in range(audio.shape[0]):
        filtered_audio[channel] = signal.sosfilt(sos, audio[channel])
    
    return filtered_audio

def add_subtle_mid_channel_saturation(mid, config):
    # Define saturation parameters
    saturation_amount = 0.03  # Very subtle, adjust as needed
    blend_factor = 0.3  # Subtle blend, adjust as needed

    # Apply saturation to the entire mid channel
    saturated_mid = np.tanh(mid * (1 + saturation_amount)) / (1 + saturation_amount)

    # Blend the saturated mid with the original mid
    mid_enhanced = mid * (1 - blend_factor) + saturated_mid * blend_factor

    return mid_enhanced

def apply_peaking_filter(signal, freq, q, gain_db, sample_rate):
    w0 = 2 * np.pi * freq / sample_rate
    alpha = np.sin(w0) / (2 * q)
    A = 10 ** (gain_db / 40)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = [b0, b1, b2]
    a = [a0, a1, a2]

    return signal.filtfilt(b, a, signal, padlen=len(signal)-1)

def calculate_average_spectrum(audio, sample_rate, fft_size):
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

def calculate_improved_rms(audio, sample_rate, config):
    def rms(x):
        return np.sqrt(np.mean(np.square(x)) + config.epsilon)

    # Define piece size (3 seconds)
    piece_size = 3 * sample_rate
    
    # Divide audio into pieces
    pieces = np.array_split(audio, max(1, len(audio) // piece_size))

    # Calculate RMS for each piece
    rms_values = np.array([rms(piece) for piece in pieces])

    # Calculate average RMS
    avg_rms = np.mean(rms_values)

    # Identify loudest pieces (above average RMS)
    loud_mask = rms_values >= avg_rms

    # Calculate final RMS using only the loudest pieces
    final_rms = rms(np.concatenate([pieces[i] for i in range(len(pieces)) if loud_mask[i]]))

    return final_rms
