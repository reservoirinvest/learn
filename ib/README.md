# Introduction
Set of programs for Interactive Brokers - <b>Trade Order Management System (TOMS)</b>

# TODOs
* [ ] Document **TOMS**
* [ ] *dfrq* elimination in *nakeds.py*
* [ ] Add YAML `filter` selection for *nakeds.py*
* [ ] Make a `Connection` class that detects if server is up or not

## TODO: Questions for analysis
* Maximum grosspos for NSE (strike * lot). This should be the grosspos benchmark.
* Extract bid-ask-last for NSE from website. This is useful for pre-market trade setup.

# Methodology
```mermaid
graph LR
	B[Build] --> 
	C[Confirm] -->
	M[Monitor]
	style B fill:#f6d6bd, stroke:#333, stroke-width:4px
	style C fill:#c0edef, stroke:#333, stroke-width:4px
	style M fill:#eeffcc, stroke:#333, stroke-width:4px

```

## Build

* Builds a `base` model for options:
  - either with *prices* and *margins*
  - or without *prices* and *margins*
* Makes
  - `covers` for covered calls and puts SELLs
  - `defends` for defending existing positions BUYs
  - `orphans` for orphaned defenses
  - `harvests` for matured options

<ins>Note</ins>
* Build uses functions from engine.py and support.py
* `base` model can be built on **`PAPER`** account
* `covers`, `defends`, `orophans` and `harvests` are to be done on a **`LIVE`** account

## Confirm

* Visualize and evaluate OHLC unds and options
* Evaluate overall Positions, P&L and Risk scenarios 
* Adjust YAML parameters
* Deep-dive on specific symbols
* Order 

## Monitor

* Breaches
* Orders and Fills
	- Dynamic price modification upon fill
* Positions 
	- with Status (balanced, orphaned, uncovered, undefended, dodos, unharvested)
	- for Risk: Reward
* System health

# Core Functions

## 1. ENGINE FUNCTIONS:

A set of functions running on IB clientID provided

### For a market

* generates `df_symlots`

* makes `und_cts`

* from und_cts generates:
	* `df_unds.pkl`
	* `df_ohlcs.pkl`
	* `df_chains.pkl`
	
	* generates `und_margins`
		* update `df_unds` with `und_margins`

	* qualifies the options `qualify_opts` 
    	* with an option to REUSE  

## 2. DFRQ FUNCTION:

Runs independently on `clientID = 0`

Generates remaining quantities `remq` based on individual and overall position
* Gets margins consumed by portfolio
* Computes the gross positions
* Computes remaining quantities based on each gross position
* Builds statuses from portfolio:
	* Statuses are: `partial`, `nakeds`, `orphans`, `uncovered`, `undefended`, `dodos`, `balanced` and `harvest`

## 3. NAKEDS FUNCTION:

Runs independently on `clientID = 0`

* run `dfrq`

* determines standard deviations from YAML settings
	* removes `blacklists`

* integrates `fallrise`

* generate `df_opts` by:
	* loading `qopts.pkl` (or) running QUALOPTS function
	* get option `margin`, `price` and `iv` with `time_stamp`
	* separates `call` and `put` options
	* save `df_nakeds.pkl` and `df_nakeds.xlsx` 

* has options to:
	- recalculate underlyings
	- generate only for one symbol
		- useful for weekend / Friday trades on specials
	- give only the earlies DTE
	- saving in custom filenames

## 4. ORPHAN FUNCTION
* For orphaned options
	
	
## 5. COVER FUNCTION
* For the uncovered	
	
	
## 6. DEFEND FUNCTION
* For the undefended and inadequately defended


## 7. # TODO: HARVEST FUNCTION
* For ripe NAKEDs


## 8. # TODO: MONITOR FUNCTION
* Automate trades

## 9. TODO: REPORTS & ANALYSIS
* Daily trades and performance
* Slice & Dice
* Key learnings
* Wish list

# Support functions
* in support.py

## Classes
* Establish variables from YAML: `Vars` 
* Set up time measurements for core functions: `Timer`

## Functions
### Days to Expiry: `get_dte`
### Standard Deviation Multiple for a df: `calcsdmult_df`
- Typically used for `strike` price

### Standard Deviation Multiple for one price: `calcsd`

### Convert to absolute integer to prevent div-by-zero error: `abs_int`

### Check if the market is open for trades: `isMarketOpen`

### Generate portfolio quickly: `quick_pf`

### Generate P&L: `get_pnl`

# Installation Notes:

## 1. IBKR

### Use pre-defined xml configuration
* Found in `./data/template` folder
* If not found, or not working, try adjusting the following:
#### a. Set memory size
Choose 1024 MB memory allocation option as shown below, in *File -> Global Configuration*:
![memory-option](./data/img/memory_allocation.jpg)

#### b. API settings
Use the following API settings in *File -> Global Configuration* :
![api-settings](./data/img/api_settings.jpg)
* Ensure that the socket-port is aligned to your PAPER v/s LIVE strategy

#### c. Disable `Price Management Algo` in TWS
A pesky `price management algo` message which freezes the screen, appears when placing multiple orders in TWS. To disable this:

- Click on the File (or Edit) menu
- Select Global Configuration
- Select Presets followed by Stocks (or any other contract type)
- Scroll down to the Miscellaneous section and check or uncheck the box for Use Price Management Algo
- Click Apply and OK to save the change.
![api-settings](./data/img/algo_disable.jpg)

### d. Set heapsize
Edit `C:\Jts\tws.vmoptions` file to adjust the heapsize as shown in the picture below. You can find this in *Help -> About <ins>T</ins>rader workstation...* menu.

![heap-size configuration](./data/img/heap_size.jpg)

Refer [to this link](https://www.interactivebrokers.com/en/software/tws/usersguidebook/priceriskanalytics/custommemory.htm) for more details.

## 2. Jupyter Lab

### Set up Jupyter to start in the folder of choice
* Build a shortcut for Jupyter Lab
* Set the shortcut to open in folder of choice
	- Refer to [this stackoverflow post](https://stackoverflow.com/a/40514875/7978112) 

## 3. VS Code

### Extensions installed
* [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* [Juypter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
* [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments) - for colour-coding comments
* [Markdown Peview Enhanced](https://marketplace.visualstudio.com/items?itemName=shd101wyy.markdown-preview-enhanced) - also supports mermaid!
	* [Mermaid Markdown Syntax Highilighting](https://marketplace.visualstudio.com/items?itemName=bpruitt-goddard.mermaid-markdown-syntax-highlighting)
* [Peacoock](https://marketplace.visualstudio.com/items?itemName=johnpapa.vscode-peacock) - for colour-coding vs code instances
* [Color Manager](https://marketplace.visualstudio.com/items?itemName=RoyAction.color-manager) - for consistent color coding scheme from `lopsec.com Pixelart Palettes`

### * Enable `black` formatter with auotosave
   - *Note*: black has to be installed `pip install black`

* Go to Settings -> `python formatting provider` and choose `black` in it

* Alternatively set the following in .vscode-> settings.json:
>    "python.formatting.provider": "black",
>    "editor.formatOnSave": true

### * Disable pylint warning
 - `Instance has no member for class`
* Set the following in .vscode->settings.json:
> "python.linting.pylintArgs":[ "--load-plugins"] 

### * Enable organize imports on save
> "editor.codeActionsOnSave": {"source.organizeImports": true}