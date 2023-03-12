import os
import pathlib

from ib_insync import util

from covers import get_covers
from nakeds import get_nakeds
from engine import get_chains, get_ohlcs, get_symlots, get_unds, make_opts
from support import Timer, Vars, empty_trash, get_market

MARKET = get_market()

ibp = Vars(MARKET.upper())
RUN_ON_PAPER = False
SAVE = True
USE_YAML = True


# * SETUP LOGS AND CLEAR THEM
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_all.log")
util.logToFile(path=LOGFILE, level=30)
with open(LOGFILE, "w"):
    pass

# .. start the timer
all_time = Timer(f"Entire {MARKET} build")
all_time.start()

empty_trash(MARKET)

und_cts = get_symlots(MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER).contract.unique()

get_unds(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

get_ohlcs(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

get_chains(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

make_opts(MARKET, USE_YAML=USE_YAML, SAVE=True, RUN_ON_PAPER=RUN_ON_PAPER)

get_nakeds(MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER)

if MARKET == 'SNP':
    get_covers(COVERSD=ibp.COVERSD, SAVEXL=SAVE, COV_DEPTH=ibp.COV_DEPTH)

all_time.stop()
