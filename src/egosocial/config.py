import os

# absolute path to project directory (i.e. src dir)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# absolute path to assets directory (examples of imaging, face detection, etc.)
# needed for testing
ASSETS_DIR = os.path.join(PROJECT_DIR, 'assets')

CREDENTIALS_FILE = os.path.join(PROJECT_DIR, 'credentials.json.nogit')
