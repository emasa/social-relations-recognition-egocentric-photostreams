import os

# absolute path to project directory (i.e. src directory)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# temporary directory to include cache data and generated program execution 
TMP_DIR = os.path.join(PROJECT_DIR, 'tmp')

# absolute path to assets directory (examples of imaging, face detection, etc.)
# needed for testing
ASSETS_DIR = os.path.join(PROJECT_DIR, 'assets')
# logging configuration
LOGGING_CONFIG = os.path.join(PROJECT_DIR, 'logging.json')
# logging directory
LOGS_DIR = os.path.join(TMP_DIR, 'logs')

# absolute path to MCS API credentials
CREDENTIALS_FILE = os.path.join(PROJECT_DIR, 'credentials.json.nogit')

# face API url, free tier uses westcentralus servers, paid tier should use the
# closest server to get the best latency
FACE_API_BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face' \
                    '/v1.0/'

# attribute models list
MODEL_KEYS = """
BODY_AGE,BODY_GENDER,BODY_CLOTHING,
HEAD_AGE,HEAD_GENDER,HEAD_POSE,FACE_APPEARANCE,FACE_EMOTION
"""
MODEL_KEYS = [name.strip().lower() for name in MODEL_KEYS.split(',')]
# list of model's url
MODEL_FILE_URLS = [
    (
        'body_age_trained_on_pipa.h5',
        '1ZYm-KPFktPwvU6Io6E0gKToIH7ZVto5v',
        '892196f1d8440c17d9b9679af968a036576f4943',
    ),
    (
        'body_gender_trained_on_pipa.h5',
        '1R--Qp7wVZFOwYKjRrwJSV7P9dX_sE8B7',
        'bb2e474674659eb2a00110ca457879676d7cb5e4',
    ),
    (
        'body_clothing_trained_on_berkeleyBodyAttributes.h5',
        '1MtCu9x1T-p2Ob9NpXsmDyro5iPDdHSTx',
        'a05668f847efbb807cc96cff67f9b23930372e9a',
    ),
    (
        'head_age_trained_on_pipa.h5',
        '1qZ8sML37Bup4fCmRjiJr7onKIgf3gOGx',
        '2438d1b2a323027bf51d356ba2cac30c2480c010',        
    ),
    (
        'head_gender_trained_on_pipa.h5',
        '1wokz3NHzLIbH_a1AnGpUR4D_y0dCkMx7',
        '29b803e96d56b64bb1b52e6f6faf91e2f704b117',
    ),
    (
        'face_pose_trained_on_IMFDB.h5',
        '1g49QorPjkLNlNaz-Ci9RPBbpKFq8VFqb',
        'a4e5fc9c61d732c1a6d778803397d0e84c2469f1',
    ),
    (
        'face_appearance_trained_on_CelebAFaces.h5',
        '17WFPEsGNvbszFK8zsXb_HGEv_RYzhEzI',
        '05379c9c5f75e3380721d7ec7080bc8368fb145f',
    ),
    (
        'face_emotion_trained_on_IMFDB.h5',
        '1U5yNwDkPigHleh5qycdf7KYZ-VuGDSzK',
        '3b914516271619b720e768b148a0eac0e4cd6f78',
    ),
]

MODELS = {
    name: dict(zip(('file', 'file_id', 'file_hash'), MODEL_FILE_URLS[idx]))
    for idx, name in enumerate(MODEL_KEYS)
}

# directory where the models are downloaded if there aren't already there.
MODELS_CACHE_DIR = os.path.join(TMP_DIR, 'model_snapshots')
MODELS_CACHE_FULL = os.path.join(MODELS_CACHE_DIR, 'bins')
MODELS_CACHE_WEIGHTS = os.path.join(MODELS_CACHE_DIR, 'weights')
MODELS_CACHE_DEFS = os.path.join(MODELS_CACHE_DIR, 'defs')

DATASETS_DIR = os.path.join(TMP_DIR, 'datasets')
COMPRESSED_FILES_DIR = os.path.join(TMP_DIR, 'zips')

SOCIAL_IMAGES_DATA = dict(
    directory=os.path.join(DATASETS_DIR, 'images'),
    file_id='1BVza2QSVWHW-CxuOAoPjm2kQ3m5VMqMB',
    compressed_file=os.path.join(COMPRESSED_FILES_DIR, 'images.zip'),
    file_hash='58a0eb2aa4d27df77eca337cab87dadae18b6cd3',
)

SOCIAL_IMAGES_SPLITS = dict(
    directory=os.path.join(DATASETS_DIR, 'splits'),
    file_id='1-oPtA9YOe_r7gMbkmy4PPziwfIyQLJzb',
    compressed_file=os.path.join(COMPRESSED_FILES_DIR, 'splits.zip'),
    file_hash='ef95ab7c85991f7b58346f442beff9f10c0ad614',
)