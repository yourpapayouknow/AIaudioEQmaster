from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt
from ..common.paths import PROFILES_DIR, SECURED_GENRES_DIR
from ..test_model import get_suggestions_for_genre
from .audio_io import load_secured_audio
from .transforms import lr_to_ms, ms_to_lr

def load_genre_profile(genre):
    profile_path = PROFILES_DIR / f"{genre}_profile.json"
    
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    
    # If the profile has a 'features' key, return its contents
    # Otherwise, return the whole profile (for new flat structure)
    return profile.get('features', profile)

def calculate_lufs(audio, sr):
    # Ensure audio is in float32 format
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Ensure audio is in the range -1.0 to 1.0
    if audio.max() > 1.0 or audio.min() < -1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Create BS.1770 meter
    meter = pyln.Meter(sr)
    
    # Ensure audio is in (samples, channels) shape
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    elif audio.shape[0] == 2 and audio.shape[1] > 2:
        audio = audio.T
    
    # Calculate integrated loudness
    loudness = meter.integrated_loudness(audio)
    return loudness

def get_model_suggestions(genre):
    return get_suggestions_for_genre(genre)

def apply_guardrails(initial_value, suggested_value):
    max_increase = initial_value * 1.25  # 25% increase limit
    min_decrease = initial_value * 0.95  # 5% decrease limit
    return max(min(suggested_value, max_increase), min_decrease)

def create_reference_from_profile(genre_profile, config):
    genre = genre_profile['genre']
    secured_genre_file = SECURED_GENRES_DIR / f"{genre}.secgnr"
    
    if not os.path.exists(secured_genre_file):
        raise FileNotFoundError(f"Secured genre file not found: {secured_genre_file}")
    
    print(f"Loading secured genre file: {secured_genre_file}")
    audio, sr = load_secured_audio(secured_genre_file, config)
    
    # Ensure the audio is stereo
    if audio.ndim == 1:
        audio = np.tile(audio, (2, 1))
    elif audio.shape[0] > 2:
        audio = audio[:2]
    
    # Downsample by factor of 4
    target_sr = sr // 4
    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    
    # Get model suggestions
    model_suggestions = get_model_suggestions(genre)
    print(f"Model suggestions for {genre}:")
    print(f"RMS Mid: {model_suggestions['rms_mid']:.4f}")
    print(f"RMS Side: {model_suggestions['rms_side']:.4f}")
    print(f"Stereo Width: {model_suggestions['stereo_width']:.4f}")
    
    # Calculate Initial RMS
    mid, side = lr_to_ms(audio)
    initial_rms_mid = np.sqrt(np.mean(mid**2))
    initial_rms_side = np.sqrt(np.mean(side**2))
    print(f"Initial RMS - Mid: {initial_rms_mid:.4f}, Side: {initial_rms_side:.4f}")
    
    # Apply guardrails
    suggested_rms_mid = apply_guardrails(initial_rms_mid, model_suggestions['rms_mid'])
    suggested_rms_side = apply_guardrails(initial_rms_side, model_suggestions['rms_side'])
    print(f"After guardrails - RMS Mid: {suggested_rms_mid:.4f}, RMS Side: {suggested_rms_side:.4f}")
    
    # Adjust RMS
    mid_factor = suggested_rms_mid / initial_rms_mid
    side_factor = suggested_rms_side / initial_rms_side
    
    adjusted_mid = mid * mid_factor
    adjusted_side = side * side_factor
    
    adjusted_audio = ms_to_lr(adjusted_mid, adjusted_side)
    
    # Calculate final RMS values
    final_mid, final_side = lr_to_ms(adjusted_audio)
    final_rms_mid = np.sqrt(np.mean(final_mid**2))
    final_rms_side = np.sqrt(np.mean(final_side**2))
    print(f"Final RMS - Mid: {final_rms_mid:.4f}, Side: {final_rms_side:.4f}")
    
    return adjusted_audio, target_sr

def boost_band(audio, sample_rate, low_cutoff, high_cutoff, gain, order=4):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    filtered_signal = signal.sosfilt(sos, audio)
    return audio + (filtered_signal * (gain - 1))

def high_shelf_boost(audio, sample_rate, cutoff_freq, gain, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    sos = signal.butter(order, normal_cutoff, btype='highpass', output='sos')
    filtered_signal = signal.sosfilt(sos, audio)
    return audio + (filtered_signal * (gain - 1))

def low_shelf_tighten(audio, sample_rate, cutoff_freq, gain, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    sos = signal.butter(order, normal_cutoff, btype='lowpass', output='sos')
    filtered_signal = signal.sosfilt(sos, audio)
    return audio * gain + filtered_signal * (1 - gain)

def apply_eq_style(mid, side, sample_rate, eq_style):
    print(f"Applying {eq_style} EQ style")
    if eq_style == "Warm":
        # Mid channel processing
        mid = boost_band(mid, sample_rate, low_cutoff=200, high_cutoff=300, gain=1.19, order=4)  # +1.5dB
        mid = boost_band(mid, sample_rate, low_cutoff=2000, high_cutoff=3000, gain=0.89, order=4)  # -1dB
        
        # Side channel processing
        side = boost_band(side, sample_rate, low_cutoff=3500, high_cutoff=4500, gain=0.92, order=4)  # -0.7dB
        side = boost_band(side, sample_rate, low_cutoff=150, high_cutoff=210, gain=1.06, order=4)  # +0.5dB

    elif eq_style == "Bright":
        # Mid channel processing
        mid = boost_band(mid, sample_rate, low_cutoff=2700, high_cutoff=3300, gain=1.19, order=4)  # +1.5dB
        mid = boost_band(mid, sample_rate, low_cutoff=500, high_cutoff=600, gain=1.08, order=4)  # +0.7dB
        
        # Side channel processing
        side = boost_band(side, sample_rate, low_cutoff=200, high_cutoff=300, gain=0.92, order=4)  # -0.7dB
        side = high_shelf_boost(side, sample_rate, cutoff_freq=8000, gain=1.19, order=4)  # +1.5dB

    elif eq_style == "Fusion":
        # Combination of both Warm and Bright
        # Mid channel processing
        mid = boost_band(mid, sample_rate, low_cutoff=200, high_cutoff=300, gain=1.10, order=4)  # Moderate low boost
        mid = boost_band(mid, sample_rate, low_cutoff=2700, high_cutoff=3300, gain=1.15, order=4)  # Boost similar to bright

        # Side channel processing
        side = boost_band(side, sample_rate, low_cutoff=200, high_cutoff=300, gain=0.97, order=4)  # Slight cut
        side = high_shelf_boost(side, sample_rate, cutoff_freq=8000, gain=1.12, order=4)  # Slight high-end boost

    print(f"After EQ - Mid max: {np.max(np.abs(mid)):.4f}, Side max: {np.max(np.abs(side)):.4f}")
    return mid, side
