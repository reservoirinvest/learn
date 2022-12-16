# Program logs to console and two files (msg.log and all.log) based on logging levels and module name
# Using a Filter, INFO is deliberately supressed, though the level set for the logger is DEBUG
#  - all.log has all the messages
#  - console and msg.log shows INFO level and above, except for the ib_insync info

import logging


class NoInfoFilter(logging.Filter):
    def filter(self, record):
        return not record.levelname == 'INFO' # suppresses INFO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # It is important to set this to the lowest required level!!

# Prepare a format
fmt = logging.Formatter("%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s")

# Instantiate the filter
f = NoInfoFilter()

# Set up a console handler
ch = logging.StreamHandler() # create a console handler
ch.setLevel(logging.INFO) # Set its level
ch.setFormatter(fmt) # Set its format
ch.addFilter(f) # add the filter

# Set up the message file handler
fh_msg = logging.FileHandler(filename='./msg.log') # put the path
# Set msg.log's level, format and filter similar to console
fh_msg.setLevel(ch.level)
fh_msg.setFormatter(ch.formatter)
fh_msg.addFilter(f) # cannot extract class instance of filter from ch!

# Set up the all file handler
fh_all = logging.FileHandler(filename='./all.log') # put the path
fh_all.setFormatter(fmt) # set its format

logger.addHandler(ch)  # to console
logger.addHandler(fh_msg) # to msg.log
logger.addHandler(fh_all) # to all.log

if __name__ == "__main__":
    logger.warning("This is a warning message from main")
    logger.info("This is an info from main. Should log to console and message")
    logger.debug("This is a debug statement! Should log only to all.log")
