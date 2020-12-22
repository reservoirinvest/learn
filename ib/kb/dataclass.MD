# `@dataclass` for a scrip

**FUNCTIONS**: engine() (with symlots(), ohlcs(), chains(), unds(), price()) -> all_opts()

unds.symlots: pd.DataFrame
unds.ohlcs: pd.DataFrame
unds.chains: pd.DataFrame
unds.unds: pd.DataFrame

scrip: pd.DataFrame = all_opts(scrip)

  - scrip.symlot: pd.DataFrame
  - scrip.ohlcs: pd.DataFrame
  - scrip.chains: pd.DataFrame
  - scrip.unds: pd.DataFrame
  - scrip.undPrice: float
  - scrip.iv: float
  - scrip.margin: float
  - scrip.bid: float
  - scrip.ask: float
  - scrip.price: float
  - scrip.price_time: datetime.datetime

**FUNCTIONS**: dfrq(), pnl(), 

  - scrip.pf: pd.DataFrame

  - scrip.status: str
  - scrip.remq: int
  - scrip.pnl: float
  - scrip.totalMargin: float
  - scrip.rom: float
  - scrip.prop: float

**FUNCTIONS**: opts(), get_opt(), get_ohlcs(days), price(), best_opts(), expPrice()

scrip.opts : pd.DataFrame

  - scrip.opt: pd.DataFrame = scrip.opts.get_opt(strike, right, expiry, dte | nearest_dte)
  - scrip.opt.localSymbol: str
  - scrip.opt.strike: float
  - scrip.opt.right: str
  - scrip.opt.expiry: str
  - scrip.opt.dte: int
  - scrip.opt.price: float
  - scrip.opt.price_time: datetime.datetime
  - scrip.opt.ohlc: pd.DataFrame = scrip.opt.get_ohlcs(days: int=10)
  - scrip.opt.bid: float
  - scrip.opt.ask: float
  - scrip.opt.iv: float
  - scrip.opt.sdMult: float
  - scrip.opt.lot: float
  - scrip.opt.margin: float
  - scrip.opt.prop: float
  - scrip.opt.rom: float
  - scrip.opt.expPrice: float
  - scrip.opt.expRom: float
  
  - scrip.best_opt: pd.DataFrame = scrip.opt.get_best(dte | nearest_dte)


 
scrip.covers()
scrip.defends()
scrip.harvests()

# References
* Using `__getstate__` and `__setstate__` for [pickling dataclasses](https://stackoverflow.com/a/41754104/7978112)