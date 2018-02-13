import os

# absolute path to project directory (i.e. src dir)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# absolute path to assets directory (examples of imaging, face detection, etc.)
# needed for testing
ASSETS_DIR = os.path.join(PROJECT_DIR, 'assets')
# logging configuration
LOGGING_CONFIG = os.path.join(PROJECT_DIR, 'logging.json')

# absolute path to MCS API credentials
CREDENTIALS_FILE = os.path.join(PROJECT_DIR, 'credentials.json.nogit')

# face API url, free tier uses westcentralus servers, paid tier should use the
# closest server to get the best latency
FACE_API_BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face' \
                    '/v1.0/'

MODEL_KEYS = """
BODY_AGE,BODY_GENDER,BODY_CLOTHING,
HEAD_AGE,HEAD_GENDER,HEAD_POSE,FACE_APPEARANCE,FACE_EMOTION
"""
MODEL_KEYS = [name.strip() for name in MODEL_KEYS.split(',')]

MODEL_FILE_URLS = [
    ('body_age_trained_on_pipa.h5', None),
    ('body_gender_trained_on_pipa.h5', None),
    ('body_clothing_trained_on_berkeleyBodyAttributes.h5', None),
    ('head_age_trained_on_pipa.h5', None),
    ('head_gender_trained_on_pipa.h5', None),
    ('face_pose_trained_on_IMFDB.h5', None),
    ('face_appearance_trained_on_CelebAFaces.h5', None),
    ('face_emotion_trained_on_IMFDB.h5', None),
]

MODELS = {name: MODEL_FILE_URLS[idx] for idx, name in enumerate(MODEL_KEYS)}

MODELS_CACHE_DIR = os.path.join(PROJECT_DIR, 'model_snapshots', 'bins')