# Setup logs with filter

import logging
import logging.config


class no_ib_wrapper(logging.Filter):
    '''Removes ib_wrapper INFOs from log'''
    def filter(self, record):
        # return not (record.levelname == 'INFO') & ('ib_insync' in record.name)
        return not (record.levelname == 'INFO') & any(x in 'ib_insync' for x in record.name)

LOGGING = {
    'version': 1,
    'filters': {
    'no_ib_wrapper': {
         '()': no_ib_wrapper}
        },
    'loggers': {
        '': {  # root logger
            'level': 'NOTSET',
            'handlers': ['debug_console_handler', 'info_file_handler', 'error_file_handler'],
        }
    },
    'handlers': {
        'debug_console_handler': {
            'level': 'INFO',
            'formatter': 'info',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'filters': ['no_ib_wrapper']
        },
        'info_file_handler': {
            'level': 'INFO',
            'formatter': 'info',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': './data/log/info.log',
            'mode': 'a',
            'maxBytes': 1048576,
            'backupCount': 10
        },
        'error_file_handler': {
            'level': 'ERROR',
            'formatter': 'error',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': './data/log/error.log',
            'mode': 'a',
            'maxBytes': 1048576,
            'backupCount': 10           
        }
    },
    'formatters': {
        'info': {
            'format': '%(asctime)-15s - %(name)-5s - %(levelname)-8s - %(message)s'
        },
        'error': {
            'format': "%(levelname)s <PID %(process)d:%(processName)s> %(name)s.%(funcName)s(): %(message)s"
        },
    }
}
    
logging.config.dictConfig(LOGGING)

if __name__ == "__main__":
    
    import yaml
    from support import connect

    MARKET = 'SNP'

    # Initialize variables
    with open('var.yml') as f:
        data=yaml.safe_load(f)

    HOST = data["COMMON"]["HOST"]
    PORT = data[MARKET.upper()]["PORT"]
    CID = data["COMMON"]["CID"]
    LOGPATH = data[MARKET.upper()]["LOGPATH"]
    
    logging.config.dictConfig(LOGGING)
    logging.debug('hello')
    logging.debug('hello - noshow')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info('This is an info message')
    logger.error('This is an error message')
    
    with connect(HOST, PORT, CID) as ib:
        print(ib.isConnected())




""" def logToFile(path, level=logging.INFO):
    # Create a log handler that logs to the given file with a filter
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)-15s - %(name)-5s - %(levelname)-8s - %(message)s')
    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)
    f = no_wrapper_msg()
    handler.addFilter(f)
    logger.addHandler(handler)

def logToConsole(level=logging.INFO):
    # Create a log handler that logs to the console.
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)-15s - %(name)-5s - %(levelname)-8s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(filter)
    # logger.addFilter(filter)
    logger.handlers = [
        h for h in logger.handlers
        if type(h) is not logging.StreamHandler]
    logger.addHandler(handler) """    
