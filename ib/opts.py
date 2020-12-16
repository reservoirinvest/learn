# Builds option pickles
import os
import pathlib

import pandas as pd

from ib_insync import util

from engine import (
    get_chains,
    get_ohlcs,
    get_opts,
    get_symlots,
    get_unds,
    opt_margins,
    opt_prices,
    qualify_opts,
)
from support import Timer, empty_trash, get_market, yes_or_no

# get the market
MARKET = get_market()

# .. start the timer
all_time = Timer("Engine")
all_time.start()

RUN_ALL = yes_or_no(f"\n Run ALL base for {MARKET}? ")

DELETE_FILES = yes_or_no(f"\n Delete previous pickles and xlsx? ")

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")

# * SETUP LOGS AND CLEAR THEM
LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_opts.log")
util.logToFile(path=LOGFILE, level=30)
with open(LOGFILE, "w"):
    pass

# Initialize
RUN_BASE = False

if DELETE_FILES:
    empty_trash(MARKET)

if RUN_ALL:

    RUN_BASE = RUN_QUALIFY = RUN_PRICE = RUN_MARGIN = FINAL_OPTS = True

    RUN_ON_PAPER = yes_or_no(f"\n Build all base on paper for {MARKET}? ")
    REUSE = yes_or_no(f"\n Reuse qualify, price and margin for {MARKET}? ")

else:

    RUN_QUALIFY = yes_or_no(f"\n Qualify opts? ")
    RUN_PRICE = yes_or_no(f"\n Run price? ")
    RUN_MARGIN = yes_or_no(f"\n Run margin? ")

    if RUN_QUALIFY:

        msg = "qualify"

        if RUN_PRICE:
            msg = "qualify and price"
        elif RUN_MARGIN:
            msg = "qualify and margin"
        elif RUN_PRICE & RUN_MARGIN:
            msg = "qualify, price and margin"
    else:

        msg = "price and margin"

        if RUN_PRICE and not RUN_MARGIN:
            msg = "price"

        if RUN_MARGIN and not RUN_PRICE:
            msg = "margin"

    if RUN_QUALIFY | RUN_PRICE | RUN_MARGIN:
        REUSE = yes_or_no(f"\n Reuse {msg} for {MARKET}? ")

    FINAL_OPTS = yes_or_no(f"\n Assemble final df_opts for {MARKET}? ")

    if RUN_QUALIFY:
        RUN_ON_PAPER = yes_or_no(f"\n Qualify options for {MARKET} from PAPER? ")

    if RUN_PRICE:
        RUN_ON_PAPER = yes_or_no(f"\n Get option prices for {MARKET} from PAPER? ")

    if RUN_MARGIN:
        RUN_ON_PAPER = yes_or_no(f"\n Get option margins for {MARKET} from PAPER? ")


# * ACT ON WHAT HAS BEEN SELECTED

if RUN_BASE:
    df_symlots = get_symlots(MARKET=MARKET, RUN_ON_PAPER=RUN_ON_PAPER)

    und_cts = df_symlots.contract.unique()

    get_unds(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

    get_ohlcs(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

    get_chains(MARKET=MARKET, und_cts=und_cts, RUN_ON_PAPER=RUN_ON_PAPER, SAVE=True)

if RUN_QUALIFY:
    qualify_opts(
        MARKET=MARKET,
        BLK_SIZE=200,
        RUN_ON_PAPER=RUN_ON_PAPER,
        USE_YAML_DTE=True,
        CHECKPOINT=True,
        REUSE=REUSE,
        OP_FILENAME="qopts.pkl",
    )
    if FINAL_OPTS:
        get_opts(MARKET=MARKET, OP_FILENAME="df_opts.pkl")


if RUN_PRICE:
    opt_prices(
        MARKET=MARKET,
        RUN_ON_PAPER=RUN_ON_PAPER,
        REUSE=REUSE,
        OP_FILENAME="df_opt_prices.pkl",
    )

if RUN_MARGIN:
    opt_margins(
        MARKET=MARKET,
        RUN_ON_PAPER=RUN_ON_PAPER,
        REUSE=REUSE,
        OP_FILENAME="df_opt_margins.pkl",
    )
if FINAL_OPTS:
    get_opts(MARKET=MARKET, OP_FILENAME="df_opts.pkl")

all_time.stop()
