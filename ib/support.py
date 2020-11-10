# Common support functions

import datetime
import math

import numpy as np
from ib_insync import util


def get_dte(dt):
    """Gets days to expiry

    Arg:
        (dt) as day in string format 'yyyymmdd'
    Returns:
        days to expiry as int"""

    try:
        dte = (util.parseIBDatetime(dt) -
               datetime.datetime.utcnow().date()).days
    except Exception:
        dte = None

    return dte


def calcsdmult_df(price, df):
    '''Back calculate standard deviation MULTIPLE against undPrice for given price. Needs dataframes.

    Args:
        (price) as series of price whose sd needs to be known in float
        (df) as a dataframe with undPrice, dte and iv columns in float

    Returns:
        Series of std deviation multiple as float

        '''
    sdevmult = (price - df.undPrice) / \
        ((df.dte / 365).apply(math.sqrt) * df.iv * df.undPrice)
    return abs(sdevmult)


def calcsd(price, undPrice, dte, iv):
    '''Calculate standard deviation MULTIPLE for given price.

    Args:
        (price) the price whose sd needs to be known in float
        (undPrice) the underlying price in float
        (dte) the number of days to expiry in int
        (iv) the implied volatility in float

    Returns:
        Std deviation of the price in float

        '''
    try:
        sdev = abs((price - undPrice) / (sqrt(dte / 365) * iv * undPrice))
    except Exception:
        sdev = np.nan
    return sdev


if __name__ == "__main__":
    print(get_dte("20210101"))
