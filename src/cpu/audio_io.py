from ..lib import np, pyln, librosa, sf, signal, interpolate, lowess, os, json, io, time, numba, argparse, Pool, ThreadPoolExecutor, joblib, butter, filtfilt
from .config import KEY, xor_encrypt_decrypt

def load_secured_audio(file_path, config):
    with open(file_path, 'rb') as f:
        encrypted_data = f.read()
    decrypted_data = xor_encrypt_decrypt(encrypted_data, KEY)
    mp3_buffer = io.BytesIO(decrypted_data)
    audio, sr = sf.read(mp3_buffer)
    return audio.T, sr

def load_audio(file_path, config):
    print(f"Loading audio file: {file_path}")
    if file_path.endswith('.secgnr'):
        audio, sr = load_secured_audio(file_path, config)
    else:
        audio, sr = librosa.load(file_path, sr=None, mono=False)
    print(f"Loaded audio shape: {audio.shape}, max={np.max(np.abs(audio))}, min={np.min(np.abs(audio))}")
    return audio, sr

def save_audio(audio, file_path, sr):
    print(f"Saving audio to: {file_path}")
    print(f"Audio shape: {audio.shape}, max={np.max(np.abs(audio))}, min={np.min(np.abs(audio))}")
    print(f"Saving with sample rate: {sr}")
    sf.write(file_path, audio.T, sr, subtype='PCM_24')

def apply_dither(audio, bits=24):
    """
    Fast triangular dithering using optimized NumPy operations
    """
    # Single random operation instead of two separate ones
    noise = (2 * np.random.random(audio.shape) - 1) * (1.0 / (2**(bits-1)))
    return audio + noise
