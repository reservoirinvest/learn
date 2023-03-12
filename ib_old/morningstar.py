import aiohttp 
import asyncio
from typing import Union, List
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
import unicodedata as ud
from yahooquery import Ticker

async def async_load_url(url: str, session:aiohttp.ClientSession):

    try:
        async with session.get(url) as resp:
            content = await resp.text()
            # soup = bs(ud.normalize('NFKD', html), 'html.parser')
            # content = soup.find_all('div', attrs={'class': 'mdc-table-header__label'})
    except IOError:
        print(f"\nFailed to load {url}!!!\n")
        content = None

    return content

async def async_url_contents(urls: Union[list, str]):

    if isinstance(urls, str): 
        urls = [urls]  # convert to an iterable

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(async_load_url(u, session)) for u in urls]
        await asyncio.wait(tasks)

        return tasks

def df_scrapper(soup: requests.models.Response) -> pd.DataFrame:

    headers = soup.find_all('div', attrs={'class': 'mdc-table-header__label'})
    headers = ["".join(d.text.split()) for d in headers]

    cols_3 = soup.find_all('span', class_='mdc-data-point mdc-data-point--string')
    cols_3 = ["".join(d.text.split()) for d in cols_3]
    val_4 = soup.find_all('span', class_='mdc-data-point mdc-data-point--number')
    val_4 = ["".join(d.text.split()) for d in val_4]
    bool_1 = soup.find_all('span', class_='mdc-data-point mdc-data-point--boolean')
    bool_1 = ["".join(d.text.split()) for d in bool_1]
    d = dict()
    d[headers[0]] = cols_3[0::3]
    d[headers[1]] = cols_3[1::3]
    d[headers[2]] = cols_3[2::3]

    d[headers[3]] = val_4[0::4]
    d[headers[4]] = val_4[1::4]
    d[headers[5]] = val_4[2::4]
    d[headers[6]] = val_4[3::4]

    d[headers[7]] = bool_1

    df = pd.DataFrame(d)

    return df

def get_etfs() -> pd.DataFrame:
    """Gets 5-star ETFs from morningstar with latest price"""

    no_of_pages = 2 # needs regular checks!
    URL = "https://www.morningstar.com/5-star-etfs?page="

    urls = [URL+str(n) for n in range(1,no_of_pages+1)]

    results = asyncio.get_event_loop().run_until_complete(async_url_contents(urls))
    soups = [bs(ud.normalize('NFKD', res.result()), features="lxml") for res in results]
    dfs = [df_scrapper(s) for s in soups]
    edf = pd.concat(dfs, ignore_index=True)

    edf[edf.columns[3:7]] = edf[edf.columns[3:7]].apply(pd.to_numeric, errors='coerce')

    # rename columns
    cols_dict = {"Name": "name", "Ticker": "symbol", "MorningstarCategory": "category", "AdjustedExpenseRatio%": "xpratio", "ReturnRankinCategory1Y%": "rank_1y",
                 "ReturnRankinCategory3Y%": "rank_3y", "ReturnRankinCategory5Y%": "rank_5y", "ActiveorPassive": "managed"}
    edf.columns = edf.columns.map(cols_dict)

    edfsyms = edf.symbol.to_list()

    price_ticks = Ticker(edfsyms, asynchronous=True).price
    prices =  pd.Series({s: float(price_ticks[s]['regularMarketPrice']) for s in edfsyms}, name='price')
    edf['price'] = edf.symbol.map(prices)

    return edf
    
if __name__ == '__main__':
    edf = get_etfs()

    # print the best-performing 5-star funds for 3 years!
    print(edf[edf.rank_3y <= 1])