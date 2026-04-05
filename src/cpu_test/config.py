from ..lib import os

KEY = "SimpleKey123"


def xor_encrypt_decrypt(data, key):
    return bytes(a ^ ord(key[i % len(key)]) for i, a in enumerate(data))


class Config:
    def __init__(self):
        self.threshold = 0.95
        self.knee_width = 0.1
        self.min_value = 1e-8
        self.max_piece_size = 44100 * 5
        self.internal_sample_rate = 44100
        self.lowess_frac = 0.15
        self.lowess_it = 2
        self.lowess_delta = 0.1
        self.rms_correction_steps = 4
        self.limiter_threshold = 0.98
        self.limiter_knee_width = 0.3
        self.limiter_attack_ms = 1
        self.limiter_release_ms = 1500
        self.reference_file = os.path.join(os.path.dirname(__file__), "reference_track.mp3")
        self.fft_size = 2048
        self.lin_log_oversampling = 2
        self.clipping_threshold = 0.99
        self.clipping_samples_threshold = 8
        self.high_shelf_freq = 8000
        self.high_shelf_gain_db_mid = -1.5
        self.high_shelf_gain_db_side = -0.5
        self.lowpass_cutoff = 18000
        self.bypass_high_shelf = False
        self.genre = None
        self.oversampling_factor = 2
        self.epsilon = 1e-8
        self.bass_preservation_freq = 10
        self.bass_preservation_blend = 0.98
        self.apply_stereo_widening = True
        self.stereo_width_adjustment_factor = 0.2
        self.sample_rate = 44100
        self.loudness_option = "normal"
        self.eq_style = "Neutral"
        self.use_loudest_parts = False
        self.loudness_threshold = 0.4

