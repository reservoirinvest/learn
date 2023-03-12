# Dynamically order price change for fills
import os
import pathlib
import sys
from collections import defaultdict

from ib_insync import IB, util

from support import Vars, get_market

# * REPRICE ORDERS UPON FILLS


def onExecDetails(trade, fill, SCALE=0.25):

    # get the symbol of the trade filled
    try:
        symbol = {t.contract.symbol for t in trade}
    except TypeError:
        symbol = {trade.conctract.symbol}

    # Download open orders and open trades
    ib.reqOpenOrders()
    trades = ib.trades()

    # * TARGET DATAFRAME

    # . make the df
    df = (
        util.df(t.contract for t in trades)
        .iloc[:, :6]
        .assign(
            contract=[t.contract for t in trades],
            order=[t.order for t in trades],
            status=[t.orderStatus.status for t in trades],
        )
        .join(util.df(t.order for t in trades).iloc[:, 2:7])
        .rename(columns={"lastTradeDateOrContractMonth": "expiry"})
    )

    # . filter the df
    ACTIVE_STATUS = ["ApiPending", "PendingSubmit", "PreSubmitted", "Submitted"]
    mask = (
        df.status.isin(ACTIVE_STATUS) & (df.action == "SELL") & (df.symbol.isin(symbol))
    )
    df = df[mask]

    # . set the new price
    df["newLmt"] = np.where(
        df.action == "SELL",
        df.lmtPrice + df.lmtPrice * (1 + SCALE),
        df.lmtPrice - df.lmtPrice * (1 - SCALE),
    )

    df["newLmt"] = df["newLmt"].apply(lambda x: get_prec(x, ibp.PREC))

    # * CANCEL AND RE-ORDER

    # . cancel the orders first. These gives `Error validating request for VOL` (ref: )
    cancels = [ib.cancelOrder(o) for o in df.order]

    # . change order price to new limit price
    df = df.assign(
        order=[
            LimitOrder(action=action, totalQuantity=totalQuantity, lmtPrice=newLmt)
            for action, totalQuantity, newLmt in zip(
                df.action, df.totalQuantity, df.newLmt
            )
        ]
    )

    # . build the contract, orders and re-order
    cos = tuple(zip(df.contract, df.order))

    modified_trades = place_orders(ib=ib, cos=cos)

    return modified_trades


def main_loop():

    MARKET = get_market()

    ibp = Vars(MARKET.upper())  # IB Parameters from var.yml

    HOST, PORT, CID = ibp.HOST, ibp.PORT, ibp.MASTERCID

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    LOGPATH = pathlib.Path.cwd().joinpath(THIS_FOLDER, "data", "dynamic")

    # * SETUP LOGS AND CLEAR THEM
    LOGFILE = LOGPATH.joinpath(MARKET.lower() + "_nakeds.log")
    util.logToFile(path=LOGFILE, level=30)
    with open(LOGFILE, "w"):
        pass

    # * SET THE CONNECTION
    try:
        ib.isConnected()
    except NameError:
        ib = IB().connect(HOST, PORT, CID)

    while 1:
        ib.execDetailsEvent += onExecDetails
        ib.sleep(2)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nExiting by user request.\n")
        sys.exit(0)
