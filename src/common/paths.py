from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
RESOURCES_DIR = PROJECT_ROOT / 'resources'
MODEL_DIR = RESOURCES_DIR / 'model'
PROFILES_DIR = RESOURCES_DIR / 'profiles'
SECURED_GENRES_DIR = RESOURCES_DIR / 'secured_genres'
