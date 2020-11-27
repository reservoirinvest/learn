# Builds option pickles
import os
import pathlib

import pandas as pd
from ib_insync import util

from engine import (get_chains, get_ohlcs, get_opts, get_symlots,
                    get_und_margins, get_unds, opt_margins, opt_prices,
                    qualify_opts)
from support import Timer, get_market, yes_or_no

# get the market
MARKET = get_market()

# .. start the timer
all_time = Timer("Engine")
all_time.start()

RUN_ALL = yes_or_no(f"\n Run ALL base for {MARKET}? ")

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")

# * SETUP LOGS AND CLEAR THEM
LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_opts.log")
util.logToFile(path=LOGFILE, level=30)
with open(LOGFILE, "w"):
    pass


if not RUN_ALL:

    RUN_QUALIFY = yes_or_no(f"\n Qualify opts? ")

    RUN_PRICE = yes_or_no(f"\n Run price? ")
    RUN_MARGIN = yes_or_no(f"\n Run margin? ")

    msg = 'price' if RUN_PRICE else 'margin'

    REUSE = yes_or_no(f"\n Reuse {msg} for {MARKET}? ")

    ASSEMBLE_OPTS = yes_or_no(f"\n Assemble final df_opts for {MARKET}? ")

else:

    RUN_QUALIFY = RUN_PRICE = RUN_MARGIN = ASSEMBLE_OPTS = True

    BASE_ON_PAPER = yes_or_no(f"\n Build base on paper for {MARKET}? ")
    REUSE = yes_or_no(f"\n Reuse price / margin for {MARKET}? ")

    df_symlots = get_symlots(MARKET=MARKET, RUN_ON_PAPER=BASE_ON_PAPER)
    und_cts = list(df_symlots.contract.unique())

    get_unds(MARKET=MARKET, und_cts=und_cts,
             RUN_ON_PAPER=BASE_ON_PAPER, savedf=True)
    get_und_margins(MARKET=MARKET, und_cts=und_cts,
                    RUN_ON_PAPER=BASE_ON_PAPER, savedf=True)
    get_ohlcs(MARKET=MARKET, und_cts=und_cts,
              RUN_ON_PAPER=BASE_ON_PAPER, savedf=True)
    get_chains(MARKET=MARKET, und_cts=und_cts,
               RUN_ON_PAPER=BASE_ON_PAPER, savedf=True)


if RUN_QUALIFY:
    qualify_opts(MARKET=MARKET, BLK_SIZE=500, RUN_ON_PAPER=BASE_ON_PAPER,
                 CHECKPOINT=True, OP_FILENAME="qopts.pkl")

if RUN_PRICE:
    PRICE_ON_PAPER = yes_or_no(
        f"\n Get option prices for {MARKET} from PAPER? ")
    opt_prices(MARKET=MARKET, RUN_ON_PAPER=PRICE_ON_PAPER,
               REUSE=REUSE, OP_FILENAME='df_opt_prices.pkl')

if RUN_MARGIN:
    MARGIN_ON_PAPER = yes_or_no(
        f"\n Get option margins for {MARKET} from PAPER? ")
    opt_margins(MARKET=MARKET, RUN_ON_PAPER=MARGIN_ON_PAPER,
                REUSE=REUSE, OP_FILENAME='df_opt_margins.pkl')
if ASSEMBLE_OPTS:
    get_opts(MARKET=MARKET, OP_FILENAME='df_opts.pkl')

all_time.stop()
