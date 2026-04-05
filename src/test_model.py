import json
from .lib import joblib, np
from .common.paths import MODEL_DIR, PROFILES_DIR

FOCUS_GENRES = ['Pop','EDM','Rock','Dance','Hiphop','Ambient','Chillout','Orchestral','Speech','Piano']

def load_genre_model():
    model_path = MODEL_DIR / 'genre_model.joblib'
    feature_scaler_path = MODEL_DIR / 'genre_feature_scaler.joblib'
    target_scaler_path = MODEL_DIR / 'genre_target_scaler.joblib'
    if not (model_path.exists() and feature_scaler_path.exists() and target_scaler_path.exists()):
        raise FileNotFoundError('Model or scalers not found')
    return joblib.load(model_path), joblib.load(feature_scaler_path), joblib.load(target_scaler_path)

def prepare_input_features(profile):
    numerical_features=[profile['initial_rms'],profile['rms_mid'],profile['rms_side'],profile['rms_after_matching'],profile['stereo_width_mid'],profile['stereo_width_side'],profile['lufs'],profile['spectral_centroid'],profile['spectral_bandwidth']]
    max_val=max(numerical_features); min_val=min(numerical_features)
    normalized=[(x-min_val)/(max_val-min_val) for x in numerical_features]
    genre_encoding=[1 if profile['genre']==g else 0 for g in FOCUS_GENRES]
    features=normalized+genre_encoding+profile['simplified_spectrum_mid']+profile['simplified_spectrum_side']
    return np.array(features).reshape(1,-1)

def get_suggestions_for_genre(genre):
    model, feature_scaler, target_scaler = load_genre_model()
    with open(PROFILES_DIR / f'{genre}_profile.json','r',encoding='utf-8') as f:
        profile=json.load(f)
    scaled=feature_scaler.transform(prepare_input_features(profile))
    s=target_scaler.inverse_transform(model.predict(scaled).reshape(1,-1))[0]
    return {'rms_mid':s[0],'rms_side':s[1],'stereo_width':s[2]}
