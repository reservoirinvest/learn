# This program is not working!!
# Ref: [Youtube]('https://youtu.be/Xbl7XjFYsN4?t=748')


import time
import threading
from typing import Mapping # Mapping did not work in vscode!!!
# import typing
class Mayhem(threading.Thread):
    def __init__(self, map: Mapping[str, int]) -> None:
        super().__init__()
        self.map = map
        def run(self):
            for key, value in self.map.items():
                print(f'sleep value is {value}')
                time.sleep(value)
                
d = {"k1": 1, "k2": 2, "k3": 3}
m = Mayhem(d)
m.start()
d['k4'] = 4 # This is expected to throw an error, which it doesn't !!!
print(d)