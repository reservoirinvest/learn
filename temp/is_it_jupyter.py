import os
import IPython as ipy

# string sources only
sources = str(os.environ.keys()) + \
          ipy.get_ipython().__class__.__name__

# make pattern of unique keys
checks = {'SPYDER': 'Spyder', 'QTIPYTHON': 'qt IPython', 'VSCODE': 
          'VS Code', 'ZMQINTERACTIVEshell': 'Jupyter', }

results = []
msg = []

for k, v in checks.items():
    u = str(k.upper())
    if u in sources.upper():
        results.append(checks[k])

if not results:
    msg.append("Unknown IDE")
else:
    msg.append("Program working ")
    while results:
        msg.append(f"in {results.pop()}")
        if results:
            msg.append(' with')

print(''.join(msg))