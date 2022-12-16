# Builds base data

# Prevent spurious problems for try...
# pyright: reportUnboundVariable=false

import os
import pathlib

import pandas as pd
from ib_insync import util

from engine import get_chains, get_ohlcs, get_symlots, get_unds
from opts import make_opts
from support import Timer, empty_trash, get_market, yes_or_no

# get the market
MARKET = get_market()

# Initialize
RUN_BASE = RUN_QUALIFY = NO_ACTION = False

# .. start the timer
all_time = Timer("Base")
all_time.start()

RUN_ALL = yes_or_no(f"\n Run ALL for {MARKET}?: ")

if RUN_ALL:
    RUN_BASE = RUN_QUALIFY = True
    RUN_ON_PAPER = yes_or_no(f"\n Run ALL on paper for {MARKET}?: ")

else:
    RUN_BASE = yes_or_no(f"\n Run ONLY base for {MARKET}?: ")

    if RUN_BASE:
        RUN_ON_PAPER = yes_or_no(f"\n Build base on paper for {MARKET}?: ")
    else:
        RUN_QUALIFY = yes_or_no(f"\n Run ONLY qualify for {MARKET}?: ")

        if RUN_QUALIFY:
            RUN_ON_PAPER = yes_or_no(f"\n Qualify on paper for {MARKET}?: ")
        else:
            NO_ACTION = True

if RUN_QUALIFY:
    USE_YAML = yes_or_no(f"\n Use YAML settings to qualify for {MARKET}?: ")

DELETE_FILES = yes_or_no(f"\n Delete previous pickles and xlsx?: ")

if not DELETE_FILES and NO_ACTION:
    print(f"\n\n... No action was chosen ...BYE...\n\n")

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")

# * SETUP LOGS AND CLEAR THEM
LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_opts.log")
util.logToFile(path=LOGFILE, level=30)
with open(LOGFILE, "w"):
    pass

if DELETE_FILES:
    empty_trash(MARKET)

# * ACT ON WHAT HAS BEEN SELECTED

if RUN_BASE:
    df_symlots = get_symlots(MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER)

    und_cts = df_symlots.contract.unique()

    get_unds(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

    get_ohlcs(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

    get_chains(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

if RUN_QUALIFY:
    df_opts = make_opts(MARKET, USE_YAML=USE_YAML, SAVE=True, RUN_ON_PAPER=RUN_ON_PAPER)


all_time.stop()
