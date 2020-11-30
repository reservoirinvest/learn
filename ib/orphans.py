# proposals to handle orphans

import math
import os
import pathlib
import pickle

import pandas as pd
from ib_insync import IB, MarketOrder, Option, util

from dfrq import get_dfrq
from engine import get_chains, get_margins, get_prices, get_unds, qualify
from support import (Timer, Vars, calcsdmult_df, get_dte, get_prec, get_prob,
                     quick_pf)


# Get orphans
def get_orphans(MARKET: str='SNP',
                ORPHSD_CUTOFF: float=1.4,
                ORPH_DEPTH: int=4,
                SAVEXL: bool=True) -> pd.DataFrame:
    
    orph_time = Timer(f"{MARKET} orphans time")
    orph_time.start()

    # * SETUP
    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.CID
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "log")
    DATAPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", MARKET.lower())

    # ...setup logs and clear them
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_orphans.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # ... file loads
    df_symlots = pd.read_pickle(DATAPATH.joinpath('df_symlots.pkl'))
    dfrq = pd.read_pickle(DATAPATH.joinpath('dfrq.pkl'))

    # * GET THE ORPHANED STOCKS FROM DFRQ
    dfrq = get_dfrq(MARKET)

    # ... get the portfolio, chains and unds for orphaned stocks
    with IB().connect(HOST, PORT, CID) as ib:
        df_pf = quick_pf(ib)

    orph = dfrq[dfrq.status.isin(['orphan'])].symbol.unique()

    orph_cts = set(df_symlots[df_symlots.symbol.isin(orph)].contract.unique())

    df_ch = get_chains(MARKET=MARKET, und_cts=orph_cts, savedf=False, RUN_ON_PAPER=False)
    df_u = get_unds(MARKET=MARKET, und_cts=orph_cts, savedf=False, RUN_ON_PAPER=False)

    # * BUILD THE OPTION CONTRACTS

    # ... get the next week (after 5 days) chains
    m = df_ch.dte > 5
    min_dte = df_ch[m].groupby('symbol').dte.transform(min)

    df_ch1 = df_ch[m][df_ch[m].dte == min_dte].drop('exchange', 1)

    # ... integrate iv and portfolio
    df_ch2 = df_ch1.set_index('symbol')\
        .join(df_u[['symbol', 'undPrice', 'iv']]
            .set_index('symbol')).rename(columns={'iv': 'und_iv'})

    # ... integrate portfolio and rename position as qty
    m = (df_pf.secType == 'OPT') & (df_pf.symbol.isin(orph))

    # ... keep the original strike of the option as old_strike
    df_pf1 = df_pf[m][['symbol', 'position', 'avgCost', 'strike', 'right', 'expiry']]\
            .rename(columns={'strike': 'old_strk', 'expiry': 'old_exp'})

    df_ch3 = df_ch2.join(df_pf1.set_index('symbol')) \
                .reset_index().rename(columns={'position': 'qty'})

    df_ch3['old_dte'] = df_ch3.old_exp.apply(get_dte)

    df_ch3['old_sdMult'] = calcsdmult_df(df_ch3.old_strk, df_ch3.assign(iv=df_ch3.und_iv, dte=df_ch3.old_dte))

    # ... all orphans need to be SELL only
    df_ch3['action'] = 'SELL'

    # ... filter orphans closest to the underlying price
    df_ch4 = df_ch3.groupby('symbol').\
        apply(lambda g: g.iloc[abs(g.strike - g.undPrice).argsort()[:ORPH_DEPTH]]).\
        reset_index(drop=True)

    # ... build the contracts
    optcon_raw = [Option(s, e, k, r, 'SMART')
                    for s, e, k, r
                    in zip(df_ch4.symbol, df_ch4.expiry,
                            df_ch4.strike, df_ch4.right)]

    with IB().connect(HOST, PORT, CID) as ib:
        dfoc = ib.run(qualify(ib, optcon_raw))

    dfoc = util.df(dfoc.to_list()).iloc[:, :6]\
        .rename(columns={'lastTradeDateOrContractMonth': 'expiry'})\
        .assign(contract=dfoc).dropna()\
        .reset_index(drop=True)

    # ... integrate with the chains
    cols = ['symbol', 'strike', 'expiry', 'right']
    df_ch5 = df_ch4.set_index(cols).join(
        dfoc[cols + ['conId', 'contract']].set_index(cols)).dropna().reset_index()

    # ... get the prices
    dfop = get_prices(MARKET=MARKET, cts=df_ch5.contract, FILL_DELAY=8)

    # ... get margins
    orders = [MarketOrder("SELL", qty) for qty in df_ch5.qty]

    cos = list(zip(df_ch5.contract, orders))

    dfom = get_margins(MARKET, cos, msg='orphan margins')

    # ... integrate price and margins
    df_ch6 = df_ch5.set_index('conId')\
        .join(dfop[['conId', 'bid', 'ask', 'close', 'price', 'iv']]
                .set_index('conId'))\
        .join(dfom[['conId', 'margin', 'comm']]
                .set_index('conId'))\
        .reset_index()
        
    # ... replace iv with und_iv where it is not available
    df_ch7 = df_ch6.assign(iv=df_ch6.iv.fillna(df_ch6.und_iv))

    # Set the expected price
    df_ch7['expPrice'] = (df_ch7.price + 3 *
                            ibp.PREC).apply(lambda x: get_prec(x, ibp.PREC))
    df_ch7 = df_ch7.assign(sdmult=calcsdmult_df(df_ch7.strike, df_ch7))
    df_ch7 = df_ch7.assign(prop=df_ch7.sdmult.apply(get_prob))

    # change conId float -> int
    df_ch8 = df_ch7.assign(conId=df_ch7.conId.astype('int'))

    # Stage the orphan columns
    cols = ['conId', 'contract', 'symbol', 'right', 'strike', 'old_strk',
            'undPrice', 'expiry', 'dte', 'iv', 'old_sdMult', 'sdmult',
            'lot', 'margin', 'action', 'qty', 'avgCost', 'price', 'expPrice']
    df_ch8 = df_ch8[cols]

    df_ch8 = df_ch8.assign(avgCost = df_ch8.avgCost/df_ch8.lot)

    df_ch8 = df_ch8.assign(revenue=df_ch8.price * df_ch8.lot * df_ch8.qty)

    df_ch8 = df_ch8.assign(pnl = df_ch8.revenue - df_ch8.avgCost * df_ch8.lot * df_ch8.qty)

    # Set orphans beyond sd cut-off to close and take the loss
    df_ch8.loc[df_ch8.old_sdMult > ORPHSD_CUTOFF, 'action'] = 'CLOSE'

    df_raw_orphs = df_ch8  # The entire set

    # Choose the contracts giving the highest pnl
    df_ch8 = df_ch8[df_ch8.pnl == df_ch8.groupby(
        'symbol').pnl.transform(max)].reset_index(drop=True)

    # ... report missing underlying symbols
    missing_und_symbols = [s for s in orph if s not in list(df_raw_orphs.symbol.unique())]

    if missing_und_symbols:
        print(
            f'\nNote: {missing_und_symbols} orphan options could not be addressed...\n')
        
    # ... pickle

    df_orphans = df_ch8[df_ch8.action != 'CLOSE'].reset_index(drop=True)
    df_close = df_ch8[df_ch8.action == 'CLOSE'].reset_index(drop=True)

    saveObject = {'df_orphans': df_orphans, 'df_close': df_close, 'df_raw_orphans': df_raw_orphs, 'missing': missing_und_symbols}

    with open(DATAPATH.joinpath('df_orphans.pkl'), 'wb') as f:
        pickle.dump(saveObject, f)

    if SAVEXL:
        writer = pd.ExcelWriter(DATAPATH.joinpath(
            'propose_orphans.xlsx'))
        df_orphans.to_excel(writer, sheet_name='Orphans', float_format='%.2f',
                        index=False, freeze_panes=(1, 1))
        
        df_close.to_excel(writer, sheet_name='Close', float_format='%.2f',
                        index=False, freeze_panes=(1, 1))
        
        df_raw_orphs.to_excel(writer, sheet_name='Alternatives', float_format='%.2f',
                            index=False, freeze_panes=(1, 1))
        sht1 = writer.sheets['Orphans']
        sht2 = writer.sheets['Close']
        sht3 = writer.sheets['Alternatives']
        
        for s in [sht1, sht2, sht3]:
            s.set_column('A:B', None, None, {"hidden": True})

        writer.save()
        
    orph_time.stop()
    
    return df_orphans

if __name__ == "__main__":
    get_orphans()
