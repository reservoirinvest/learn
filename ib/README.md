# Introduction
Interactive Brokers programs

# Incomplete TODOs
* [ ] Add `hi52` and `lo52` to **fresh()**
* [ ] Add YAML `filter` selection fo **fresh()**
* [ ] Make a `Connection` class that detects if server is up or not 

# Core Functions
* Usually `RUN_ON_PAPER` in the background
## 1. ENGINE FUNCTION:

### For a market

* generate `df_symlots`

* make `und_cts`

* from und_cts generate:
	* `df_unds.pkl`
	* `df_ohlcs.pkl`
	* `df_chains.pkl`
	
	* generate `und_margins`
		* update `df_unds` with `und_margins`

	* `qualify_opts` 
    	* with an option to REUSE  

## 2. DFRQ FUNCTION:
* For remaining quantities `remq` based on position and risk

## 3. FRESH FUNCTION:

* run `dfrq`

* generate `df_opts` by:
	* loading `qopts.pkl` (or) running QUALOPTS function
	* get option `margin`, `price` and `iv` with `time_stamp`
	* save `df_opts.pkl`

### For a market

* generate `dfrq`

* check `YAML FILTER`s:
	* ONLYPUTS
	* FALLRISE
	* HL52

* make df_fresh

* check how to generate:
	* from qopts.pkl? (DEFAULT)
	* from df_opts.pkl? (Fastest)
	* from raw_opts?

* scrub df_fresh
  * Generate raw_opts / load df_raw_opts
  * remove anything that is not `fresh`
  * remove `BLACKLIST`
  * remove dtes
  * ---

  * get `und_price` and `und_iv`
  * compute `1sd` or one standard deviation
  * generate the fence based on YAML call and put stdev multiples
  * remove options within the `fence`
  * ---

  * integrate `rem_qty`
  * for each dte and symbol:
  * remove options beyond ``min(rem_qty, MAXOPTQTY_SYM)`` set in YAML

* integrate `FALLRISE` and `HILO52`

* if raw_opts was selected, `qualify` option contracts if raw_opts
* get the `price`,`margin` and `iv` of the options

* set `expRom` and `expPrice`
* sort by `rom`

* split `df_calls` and `df_puts`
* if market is open, filter df_calls and df_puts for df_xx.ask > 0

* pickle `df_fresh`
* set columns and save to Excel
	
## 4. ORPHAN FUNCTION
* For orphaned options
	
	
## 5. COVER FUNCTION
* For the uncovered	
	
	
	
## 6. DEFEND FUNCTION
* For the undefended and inadequately defended



## 7. HARVEST FUNCTION
* For ripe NAKEDs



## 8. SYSTEMATIC FUNCTION
* Automate trades

## 9. REPORTS & ANALYSIS
* Daily trades and performance
* Slice & Dice
* Key learnings
* Wish list

# Supports

## Classes
### Establish variables from YAML: `Vars` 
...
### Set up time measurements for core functions: `Timer`
...

## Functions
### Days to Expiry: `get_dte`
### Standard Deviation Multiple for a df: `calcsdmult_df`
- Typically used for `strike` price

### Standard Deviation Multiple for one price: `calcsd`

### Convert to absolute integer to prevent div-by-zero error: `abs_int`

### Check if the market is open for trades: `isMarketOpen`

### Generate portfolio quickly: `quick_pf`

### Generate P&L: `get_pnl`


