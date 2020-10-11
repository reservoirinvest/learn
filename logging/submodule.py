# submodule.py
import logging

logger = logging.getLogger(__name__)

def logsomething():
    logger.info("This is an info message from submodule, should be recorded in main.log!")
    logger.debug("This is a debug message from submodule, also should be recorded in main.log!!")
    
# formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# # log to console
# c_handler = logging.StreamHandler()
# c_handler.setFormatter(formatter)
# logger.addHandler(c_handler)
