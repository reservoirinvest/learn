# * GET MAIN ACTION

import os
import platform

from support import get_market


# . determine the action needed
def get_action(act_ask):

    # process inputs
    while True:
        try:
            get_action = int('\n' + input(act_ask) + '\n')   # check for int in input
        except ValueError:
            print("\nI didn't understand what you entered. Try again!\n")
            continue  # Loop again
        if not get_action in act_ask_range:
            print(f"\nWrong number! choose between {act_ask_range}...")
        else:
            ACT = act_dict[get_action]
            break  # success and exit loop

    # clear terminal to show selection clearly
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')  # on linux and os x
        
    return ACT

# * GET THE MARKET
MARKET = get_market()

# * DETERMINE POSSIBLE ACTIONS

first_run = {0: "Quit",
             1: "Build",
             2: "Utilities"}

builds = {0: "Quit",
          1: "Build from scratch",
          2: "Build from base"
          }

nse_utils = {0: "Quit",
             1: "Harvests",
             1: "Watchlist"
             }



act_dict = {  0: "quit",
              1: "build_all",
              2: "opts_from_base",
              3: "nakeds",
              4: "harvests",
              5: "defenses",
              6: "orphans",
              7: "deletes",
              8: "watchlist"
            }



act_ask_range = list(range(len(act_dict)))
act_ask = f"\nWhat is to be done for {MARKET}? Choose from following numbers:\n"

act_ask = act_ask +   f'\n{"-" * 70}\n'
act_ask = act_ask +    "\n0) Quit the program!!\n"

act_ask = act_ask +   f'\n{"_" * 25}    1. Functions    {"_" * 25}\n'
act_ask = act_ask +      "11) Build all (df_symlots, df_unds, df_ohlcs, df_chains,\n"
act_ask = act_ask +      "               qopts, df_opt_prices, df_opt_margins, \n"
act_ask = act_ask +      "               df_opts, fresh with dfrq)\n"
act_ask = act_ask +      "12) Get underlyings (df_unds)\n"
act_ask = act_ask +      "13) Qualify options (qopts)\n"
act_ask = act_ask +      "14) Get option prices (df_opt_prices)\n"
act_ask = act_ask +      "15) Get option margins (df_opt_margins)\n"
act_ask = act_ask +      "16) Build final option set (df_opts)\n"

act_ask = act_ask + f'\n{"_" * 25}    2. Utilities    {"_" * 25}\n'

act_ask = act_ask +      "21) Get remaining quanities (dfrq)\n"
act_ask = act_ask +      "22) Build fresh nakeds (df_fresh)\n"
act_ask = act_ask +      "23) Determine the harvests (df_harvests)\n"
act_ask = act_ask +      "24) Make watchlist from symbols (watchlist.csv)\n"
act_ask = act_ask +   f'\n{"." * 70}\n'
act_ask = act_ask +     f"99) Delete ALL files for {MARKET}\n"
act_ask = act_ask +   f'\n{"." * 70}\n'

if MARKET == 'SNP':
    act_ask = act_ask + f'\n{"." * 25}       For SNP    {"." * 25}\n'
    act_ask = act_ask +  "31) Determine the covers (df_covers)\n"
    act_ask = act_ask +  "32) Determine the orphans (df_orphans)\n"
    act_ask = act_ask +  "33) Determine the defenses (df_defend)\n"

if MARKET == 'NSE':
    act_ask = act_ask + f'\n{"." * 25}       For NSE    {"." * 25}\n'
    act_ask = act_ask +  "41) Build trades for CAPSTOCKS (df_capstocks)\n"

# * GET THE ACTION
ACT = get_action(act_ask)
print(ACT)
