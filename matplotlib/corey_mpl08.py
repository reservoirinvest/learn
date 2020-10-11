# Plotting time-series data
from pathlib import Path
import pandas as pd

from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import dates as mpl_dates

register_matplotlib_converters()

p = str(Path(__file__).parent)  # file path

plt.style.use('seaborn')

# dates = [
#     datetime(2019, 5, 24),
#     datetime(2019, 5, 25),
#     datetime(2019, 5, 26),
#     datetime(2019, 5, 27),
#     datetime(2019, 5, 28),
#     datetime(2019, 5, 29),
#     datetime(2019, 5, 30)
# ]

# y = [0, 1, 3, 4, 6, 5, 7]

# plt.plot_date(dates, y, linestyle='solid', marker=None)
# plt.gcf().autofmt_xdate()  # gets the current figure
# date_format = mpl_dates.DateFormatter("%b, %d %y")
# plt.gca().xaxis.set_major_formatter(date_format)

data = pd.read_csv(p+r'\time_series_data.csv')

data['Date'] = pd.to_datetime(data.Date)
data.sort_values('Date', inplace=True)

price_date = data['Date']
price_close = data['Close']

plt.plot_date(price_date, price_close, linestyle='solid', marker=None)
plt.gcf().autofmt_xdate()  # gets the current figure

plt.title('Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.tight_layout()

plt.show()
