from ..lib import np, librosa, sf, io
from .config import KEY, xor_encrypt_decrypt


def load_secured_audio(file_path, config):
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = xor_encrypt_decrypt(encrypted_data, KEY)
    mp3_buffer = io.BytesIO(decrypted_data)
    audio, sr = sf.read(mp3_buffer)
    return audio.T, sr


def load_audio(file_path, config):
    print(f"[cpu_test] Loading audio file: {file_path}")
    if file_path.endswith(".secgnr"):
        audio, sr = load_secured_audio(file_path, config)
    else:
        audio, sr = sf.read(file_path, always_2d=True)
        audio = audio.T
    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])
    print(f"[cpu_test] Loaded shape={audio.shape}, sr={sr}")
    return audio, sr


def save_audio(audio, file_path, sr):
    print(f"[cpu_test] Saving audio to: {file_path}")
    sf.write(file_path, audio.T, sr, subtype="PCM_24")


def apply_dither(audio, bits=24):
    noise = (2 * np.random.random(audio.shape) - 1) * (1.0 / (2 ** (bits - 1)))
    return audio + noise

