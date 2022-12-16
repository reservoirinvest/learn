# main.py importing a submodule
import logging

import submodule

logger = logging.getLogger(__name__)

# log to console
c_handler = logging.StreamHandler()

console_format = logging.Formatter("[%(levelname)s] %(message)s")
c_handler.setFormatter(console_format)
c_handler.setLevel(logging.INFO)

logging.getLogger().addHandler(c_handler)

# log to file from main
logfile = "./main.log"

f_handler = logging.FileHandler(filename=logfile)

f_format = logging.Formatter("%(asctime)s: %(name)-18s [%(levelname)-8s] %(message)s")
f_handler.setFormatter(f_format)
f_handler.setLevel(logging.DEBUG)


logging.getLogger().addHandler(f_handler)
logging.getLogger().setLevel(logging.DEBUG)

logger.error("This is an error!!! Logged to console")
logger.debug("This is a debug error. Not logged to console, but should log to file")

# run submodule
submodule.logsomething()
