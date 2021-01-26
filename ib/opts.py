# generates df_opts.pkl without price and margin

import os
import pathlib
from collections import defaultdict

import pandas as pd
from ib_insync import IB, Option, util
from tqdm import tqdm

from support import (Timer, Vars, fallrise, get_dte, get_market, get_rsi,
                     yes_or_no)


def make_opts(MARKET: str, 
        USE_YAML: bool, 
        SAVE: bool, 
        RUN_ON_PAPER: bool) -> pd.DataFrame:

    # start the timer
    opt_time = Timer("Opts")
    opt_time.start()

    # set the qualified option file names
    OP_FILENAME = 'qopts.pkl'
    REJECT_FILE = OP_FILENAME[:4] + "_rejects.pkl"
    WIP_FILE = OP_FILENAME[:4] + "_wip.pkl"

    # set up local variables from YML
    ibp = Vars(MARKET)

    PORT=ibp.PAPER if RUN_ON_PAPER else ibp.PORT

    # set up paths
    cwd = pathlib.Path.cwd()
    datapath = cwd.joinpath('data', MARKET.lower())

    # log
    logfile = cwd.joinpath('data', 'log', MARKET.lower()+'_opts.log')
    util.logToFile(path=logfile, level=30)
    with open(logfile, "w"):
        pass

    # load the files
    df_chains = pd.read_pickle(datapath.joinpath('df_chains.pkl'))
    df_unds = pd.read_pickle(datapath.joinpath('df_unds.pkl'))
    df_ohlcs =  pd.read_pickle(datapath.joinpath('df_ohlcs.pkl'))

    ## ** CLEANUP

    # remove expired options
    df_chains = df_chains.assign(dte=df_chains.expiry.apply(get_dte))
    df_chains = df_chains[df_chains.dte>=0]
    df_chains.loc[df_chains.dte == 0, 'dte'] = 1 # set last day to 1 for SNP

    if USE_YAML:
        df_ch = df_chains[df_chains.dte < ibp.MAXDTE]

        # get the 6th month option for SNP defends
        if MARKET.upper() == 'SNP':
            defend_dte_dict = df_chains[df_chains.dte > eval(ibp.DEFEND_DTE)]\
                            .groupby('symbol').dte.apply(min).to_dict()
            df_ch_snp = df_chains[df_chains.dte == \
                            df_chains.symbol.map(defend_dte_dict)]
            df_ch = pd.concat([df_ch, df_ch_snp], ignore_index=True)

    else:
        df_ch = df_chains

    df_ch = df_ch.drop_duplicates()

    # build the rights
    df_ch = pd.concat([df_ch.assign(right='P'), \
                    df_ch.assign(right='C')], ignore_index=True)

    ## ** REUSE qopts

    cols = ["symbol", "expiry", "strike", "right"]

    # . qualified opts
    try:
        qopts = pd.read_pickle(datapath.joinpath(OP_FILENAME))
        qopts = qopts.drop_duplicates()

        # Clean qopts
        # ... build df_o
        df_o = (util.df(qopts.to_list()).iloc[:, :6].rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry"
            }).drop(["conId", "secType"], 1).assign(contract=qopts))

        df_o = df_o.assign(dte=df_o.expiry.apply(get_dte))
        df_o = df_o[df_o.dte>=0]

        # ... remove df_o not in df_ch
        m = df_o[cols].apply(tuple, 1).isin(df_ch[cols].apply(tuple, 1))

        qopts = df_o[m].contract.rename("qualified")

    except FileNotFoundError:
        # for existing successful options
        qopts = pd.Series([], dtype=object, name="qualified")

    # . rejected opts
    try:
        qropts = pd.read_pickle(datapath.joinpath(REJECT_FILE))
        # qropts = qropts.drop_duplicates()

        # Clean rejects
        # ... build df_rejects
        df_rejects = (util.df(qropts.to_list()).iloc[:, :6].rename(
            columns={
                "lastTradeDateOrContractMonth": "expiry"
            }).drop(["conId", "secType"], 1).assign(contract=qropts))

        df_rejects = df_rejects.assign(dte=df_rejects.expiry.apply(get_dte))
        df_rejects = df_rejects[df_rejects.dte>=0]

        # ... remove df_rejects not in df_ch
        m = df_rejects[cols].apply(tuple, 1).isin(df_ch[cols].apply(tuple, 1))

        qropts = df_rejects[m].contract.rename("rejected")

    except FileNotFoundError:
        # for rejected options
        qropts = pd.Series([], dtype=object, name="rejected")

    # . wip opts
    try:
        wips = pd.read_pickle(datapath.joinpath(WIP_FILE))
        # wips = wips.drop_duplicates()
    except FileNotFoundError:
        wips = pd.Series([], dtype=object, name="wip")

    existing_opts = pd.concat([qopts, qropts, wips], ignore_index=True)

    if not existing_opts.empty:  # something is existing!

        df_existing = util.df([e for e in existing_opts if str(e) !='nan'])\
                        .iloc[:, :6].rename(
                            columns={
                                "lastTradeDateOrContractMonth": "expiry"
                            }).drop("conId", 1)

        # Remove dte<0 for df_existing
        df_existing = df_existing.assign(dte=df_existing.expiry.apply(get_dte))
        df_existing = df_existing[df_existing.dte>=0]
        df_existing.loc[df_existing.dte == 0, 'dte'] = 1 # set last day to 1 for SNP
        df_existing = df_existing.drop_duplicates()

        # remove existing options from df_ch
        # Note: df_existing is put twice to remove it completely
        # ....  ref: so: https://stackoverflow.com/a/37313953
        df_ch = (pd.concat([df_ch[cols], df_existing[cols], df_existing[cols]],
                            ignore_index=True).drop_duplicates(
                                keep=False).reset_index(drop=True))

    # * BUILD THE OPTIONS TO BE QUALIFIED

    # imported here to prevent circular import from `base.py`
    from engine import executeAsync, post_df, qualify

    cts = [
        Option(s, e, k, r, x) for s, e, k, r, x in zip(
            df_ch.symbol,
            df_ch.expiry,
            df_ch.strike,
            df_ch.right,
            ["NSE" if MARKET.upper() == "NSE" else "SMART"] * len(df_ch),
        )
    ]

    BLK_SIZE = 200

    # ..build the raw blocks from cts
    raw_blks = [cts[i:i + BLK_SIZE] for i in range(0, len(cts), BLK_SIZE)]

    with IB().connect(ibp.HOST, PORT, ibp.QUAL) as ib:

        for b in tqdm(raw_blks,
                        desc=f"{MARKET} opts qual:",
                        bar_format=ibp.BAR_FORMAT,
                        ncols=80):

            qs = ib.run(
                executeAsync(
                    ib=ib,
                    algo=qualify,
                    cts=b,
                    CONCURRENT=200,
                    TIMEOUT=5,
                    post_process=post_df,
                    SHOW_TQDM=True,
                ))

            # Successes
            qopts = qopts.append(pd.Series(qs, dtype=object, name="qualified"),
                                    ignore_index=True)

            # Rejects
            rejects = [c for c in b if not c.conId]
            qropts = qropts.append(pd.Series(rejects,
                                                dtype=object,
                                                name="rejected"),
                                    ignore_index=True)

            if SAVE:  # store the intermediate options while qualifying
                qopts.to_pickle(datapath.joinpath(WIP_FILE))
                qropts.to_pickle(datapath.joinpath(REJECT_FILE))

        # to prevent TimeoutError()
        ib.disconnect()
        IB().waitOnUpdate(timeout=ibp.FIRST_XN_TIMEOUT)

    # Remove blanks
    qopts = pd.Series([q for q in qopts if str(q) != 'nan'], 
                        name='qualified').drop_duplicates()

    # * BUILD DF OPTION
    df_opts = (util.df(qopts.to_list()).iloc[:, :6].assign(
        contract=qopts).rename(columns={
            "lastTradeDateOrContractMonth": "expiry"
        }).drop_duplicates())

    # ... integrate lots (from chains), und_iv  and undPrice
    col1 = ["symbol", "strike", "expiry"]
    df_opts = (df_opts.set_index(col1).join(
        df_chains.set_index(col1)[["lot"]]).reset_index())

    # ... process dtes
    df_opts["dte"] = df_opts.expiry.apply(get_dte)
    df_opts = df_opts[df_opts.dte > 0]  # Remove negative dtes

    # Make 0 dte positive to avoid sqrt errors
    df_opts.loc[df_opts.dte == 0, "dte"] = 1

    df_opts["und_iv"] = df_opts.symbol.map(
        df_unds.set_index("symbol").iv.to_dict())
    df_opts["undPrice"] = df_opts.symbol.map(
        df_unds.set_index("symbol").undPrice.to_dict())

    # remove df_opts without undPrice
    df_opts = df_opts[df_opts.undPrice.notna()]

    # * INTEGRATE FALLRISE
    d = df_opts.groupby('symbol').dte.unique().to_dict()
    s = defaultdict(dict)

    for k, v in d.items():
        fr = defaultdict(dict)
        for dt in v:
            try:
                fr[dt] = fallrise(df_ohlcs[df_ohlcs.symbol==k], dt)
            except IndexError:
                pass
        s[k] = fr

    df_fr = pd.concat([pd.DataFrame([v2 for k2, v2 in v1.items()]) for k1, v1 in s.items()], ignore_index=True)

    df_opts = df_opts.set_index(['symbol', 'dte']).join(df_fr.set_index(['symbol', 'dte'])).reset_index()

    # * INTEGRATE RSI
    rsi = get_rsi(df_ohlcs)
    df_opts = df_opts.set_index('symbol').join(rsi).reset_index()

    if SAVE:
        # ... write final options and cleanup WIP
        qopts.to_pickle(datapath.joinpath(OP_FILENAME))
        df_opts.to_pickle(datapath.joinpath('df_opts.pkl'))

        try:
            os.remove(datapath.joinpath(WIP_FILE))
        except FileNotFoundError:
            pass

    opt_time.stop()

    return df_opts

if __name__ == "__main__":
    
    # user interaction
    MARKET = get_market()
    USE_YAML = yes_or_no('Filter using YAML DTE? :')
    SAVE = yes_or_no('Over-write qopts.pkl and df_opts.pkl? :')
    RUN_ON_PAPER = yes_or_no('Run on PAPER?:')

    # set up local variables from YML
    ibp = Vars(MARKET)
    locals().update(ibp.__dict__)

    df_opts = make_opts(MARKET, USE_YAML, SAVE, RUN_ON_PAPER)

    print(df_opts.head())
