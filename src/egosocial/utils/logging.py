import json
import logging.config
import os


def setup_logging(
        default_path='logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG',
        log_dir='',    
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        
        if log_dir:
            if not os.path.isdir(log_dir):
                os.mkdir(log_dir)
            
            for handler in config['handlers'].values():
                if 'filename' in handler:
                    handler['filename'] = os.path.join(log_dir, handler['filename'])
        
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
