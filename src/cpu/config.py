from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt
from ..common.paths import RESOURCES_DIR

KEY='SimpleKey123'

def xor_encrypt_decrypt(data, key):
    return bytes(a ^ ord(key[i % len(key)]) for i, a in enumerate(data))

class Config:
    def __init__(self):
        self.threshold = 0.95
        self.knee_width = 0.1
        self.min_value = 1e-8
        self.max_piece_size = 44100 * 5  # 5 seconds
        self.internal_sample_rate = 44100
        self.lowess_frac = 0.15  # Increased for more smoothing
        self.lowess_it = 2  # Increased for more robustness
        self.lowess_delta = 0.1  # Increased for faster computation
        self.rms_correction_steps = 4
        self.limiter_threshold = 0.98
        self.limiter_knee_width = 0.3
        self.limiter_attack_ms = 1
        self.limiter_release_ms = 1500
        self.reference_file = os.path.join(os.path.dirname(__file__), 'reference_track.mp3')
        self.fft_size = 4096
        self.lin_log_oversampling = 4
        self.clipping_threshold = 0.99
        self.clipping_samples_threshold = 8
        self.high_shelf_freq = 8000
        self.high_shelf_gain_db_mid = -1.5  # Reduced from -2.5
        self.high_shelf_gain_db_side = -0.5  # Reduced from -0.8
        self.lowpass_cutoff = 18000  # Changed to 18kHz
        self.bypass_high_shelf = False
        self.compressor_threshold = -3
        self.compressor_ratio = 4
        self.compressor_knee_width = 6
        self.compressor_attack_ms = 5
        self.compressor_release_ms = 50
        self.limiter_thresholds = [0.95, 0.98]
        self.limiter_knee_widths = [0.1, 0.05]
        self.limiter_attack_times = [1, 0.5]
        self.limiter_release_times = [50, 25]
        self.limiter_mix = 0.95  # Reduced limited signal mix
        self.genre = None
        self.oversampling_factor = 4  # Increased from 4 to 8
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.bass_preservation_freq = 10
        self.bass_preservation_blend = 0.98
        self.apply_stereo_widening = True  # or False
        self.stereo_width_adjustment_factor = 0.2  # Adjusts 20% of the difference by default
        self.sample_rate = 44100  # Add this line
        self.loudness_option = "normal"  # Default to "normal"
        self.eq_style = "Neutral"  # Default to "Neutral"
        self.use_loudest_parts = True
        self.loudness_threshold = 0.4
