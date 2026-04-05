from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt

@numba.jit(nopython=True)
def process_chunk(chunk, threshold, knee_width, attack_coeff, release_coeff):
    x = np.abs(chunk)
    gain_reduction = np.maximum(x / threshold, 1.0)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if threshold - knee_width / 2 < x[i, j] < threshold + knee_width / 2:
                gain_reduction[i, j] = 1.0 + ((x[i, j] - (threshold - knee_width / 2)) / knee_width) ** 2 * (x[i, j] / threshold - 1.0) / 2

    smoothed_gain = np.zeros_like(gain_reduction)
    smoothed_gain[:, 0] = gain_reduction[:, 0]

    for i in range(gain_reduction.shape[0]):
        for j in range(1, gain_reduction.shape[1]):
            if gain_reduction[i, j] > smoothed_gain[i, j-1]:
                smoothed_gain[i, j] = attack_coeff * smoothed_gain[i, j-1] + (1 - attack_coeff) * gain_reduction[i, j]
            else:
                smoothed_gain[i, j] = release_coeff * smoothed_gain[i, j-1] + (1 - release_coeff) * gain_reduction[i, j]

    return chunk / smoothed_gain

def soft_knee_compressor(audio, config):
    threshold = -6.0  # dB, slightly lower for more gentle compression
    ratio = 2.5  # Gentler ratio
    knee_width = 6.0
    attack_ms = 10.0
    release_ms = 500.0  # Longer release for smoother action

    threshold_linear = 10 ** (threshold / 20)
    
    attack_coeff = np.exp(-1 / (attack_ms * config.internal_sample_rate * config.oversampling_factor / 1000))
    release_coeff = np.exp(-1 / (release_ms * config.internal_sample_rate * config.oversampling_factor / 1000))
    
    chunk_size = 4096  # Matching the FFT size from the reference
    result = np.zeros_like(audio)
    
    for i in range(0, audio.shape[1], chunk_size):
        chunk = audio[:, i:i+chunk_size]
        compressed_chunk = process_chunk(chunk, threshold_linear, knee_width, attack_coeff, release_coeff)
        
        # Apply a compression ratio
        compressed_chunk = np.sign(compressed_chunk) * (np.abs(compressed_chunk) ** (1/ratio))
        result[:, i:i+chunk_size] = compressed_chunk
    
    return result

@numba.jit(nopython=True)
def process_multi_stage_chunk(chunk, thresholds, knee_widths, attack_coeffs, release_coeffs):
    result = chunk.copy()
    for threshold, knee_width, attack_coeff, release_coeff in zip(thresholds, knee_widths, attack_coeffs, release_coeffs):
        result = process_chunk(result, threshold, knee_width, attack_coeff, release_coeff)
    return result

def envelope_follower(x, attack_samples, release_samples):
    env = np.zeros_like(x)
    for i in range(1, x.shape[1]):
        env[:, i] = np.maximum(x[:, i], env[:, i-1] + (x[:, i] - env[:, i-1]) * (1 - np.exp(-1 / release_samples)))
    return env

@numba.jit(nopython=True, parallel=True)
def process_limiter_stage(audio, threshold, knee_width, attack_ms, release_ms, sample_rate):
    attack_samples = int(attack_ms * sample_rate / 1000)
    release_samples = int(release_ms * sample_rate / 1000)
    
    # Pre-calculate exponential terms
    attack_coeff = 1 - np.exp(-1 / attack_samples)
    release_coeff = 1 - np.exp(-1 / release_samples)
    
    # Calculate gain reduction
    gain_reduction = np.maximum(1, np.abs(audio) / threshold)
    
    # Apply knee
    knee_range = knee_width / 2
    soft_knee = np.clip((gain_reduction - (1 - knee_range)) / knee_width, 0, 1)
    gain_reduction = 1 + soft_knee**2 * (gain_reduction - 1)
    
    # Calculate smoothed gain reduction using optimized envelope follower
    smoothed_gain_reduction = np.zeros_like(gain_reduction)
    
    for i in numba.prange(audio.shape[0]):
        env = 0
        for j in range(audio.shape[1]):
            if gain_reduction[i, j] > env:
                env += (gain_reduction[i, j] - env) * attack_coeff
            else:
                env += (gain_reduction[i, j] - env) * release_coeff
            smoothed_gain_reduction[i, j] = env
    
    # Apply gain reduction only where necessary, avoiding division by zero
    epsilon = 1e-10  # Small value to prevent division by zero
    result = np.where(smoothed_gain_reduction > 1, 
                      audio / np.maximum(smoothed_gain_reduction, epsilon), 
                      audio)
    
    return result

def process_limiter_stage_with_logging(audio, threshold, knee_width, attack_ms, release_ms, sample_rate):
    result = process_limiter_stage(audio, threshold, knee_width, attack_ms, release_ms, sample_rate)
    
    print(f"Limiter stage - Threshold: {threshold}, Max input: {np.max(np.abs(audio))}")
    print(f"Max gain reduction: {np.max(result / (audio + np.finfo(audio.dtype).eps))}")
    print(f"Max output: {np.max(np.abs(result))}")
    
    return result

def multi_stage_limiter(audio, config):
     # First stage: existing implementation
    threshold1 = 10 ** (-0.6 / 20)  # -0.6 dB 
    knee_width1 = 0.1
    attack_time1 = 1.0  # ms
    release_time1 = 200.0  # ms 
    
    # Second stage: slightly more aggressive
    threshold2 = 10 ** (-0.5 / 20)  # -0.5 dB
    knee_width2 = 0.1
    attack_time2 = 3.0  # ms
    release_time2 = 900.0  # ms 
    
    sample_rate = config.internal_sample_rate * config.oversampling_factor
    
    # First stage (your existing implementation)
    result = process_limiter_stage(
        audio,
        threshold1,
        knee_width1,
        attack_time1,
        release_time1,
        sample_rate
    )
    
    # Second stage
    result = process_limiter_stage(
        result,
        threshold2,
        knee_width2,
        attack_time2,
        release_time2,
        sample_rate
    )
    
    return result
