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
