# Common support functions

import datetime

from ib_insync import util


def get_dte(dt):
    """Gets days to expiry

    Arg:
        (dt) as day in string format 'yyyymmdd'
    Returns:
        days to expiry as int"""

    try:
        dte = (util.parseIBDatetime(dt) - datetime.datetime.utcnow().date()).days
    except Exception:
        dte = None

    return dte


if __name__ == "__main__":
    print(get_dte("20210101"))
