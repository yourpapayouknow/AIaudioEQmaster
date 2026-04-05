from ..lib import np, signal


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
    nyquist = sample_rate / 2.0
    cutoff = 0.9 * nyquist
    sos = signal.butter(8, cutoff / nyquist, btype="low", output="sos")
    return signal.sosfilt(sos, audio, axis=-1)


def apply_lowpass_filter(audio, config):
    nyquist = config.internal_sample_rate * config.oversampling_factor / 2.0
    sos = signal.butter(8, config.lowpass_cutoff / nyquist, btype="low", output="sos")
    return signal.sosfilt(sos, audio, axis=-1)


def add_subtle_mid_channel_saturation(mid, config):
    saturation_amount = 0.03
    blend_factor = 0.3
    saturated_mid = np.tanh(mid * (1 + saturation_amount)) / (1 + saturation_amount)
    return mid * (1 - blend_factor) + saturated_mid * blend_factor


def calculate_improved_rms(audio, sample_rate, config):
    x = audio if audio.ndim == 1 else audio.reshape(-1)
    piece_size = int(3 * sample_rate)
    if piece_size <= 0 or len(x) <= piece_size:
        return float(np.sqrt(np.mean(np.square(x)) + config.epsilon))
    usable = (len(x) // piece_size) * piece_size
    chunks = x[:usable].reshape(-1, piece_size)
    rms_vals = np.sqrt(np.mean(chunks * chunks, axis=1) + config.epsilon)
    avg = np.mean(rms_vals)
    mask = rms_vals >= avg
    loud = chunks[mask].reshape(-1)
    if loud.size == 0:
        loud = chunks.reshape(-1)
    return float(np.sqrt(np.mean(loud * loud) + config.epsilon))

