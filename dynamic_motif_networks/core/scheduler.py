# TODO: Extend scheduler to handle asynchronous events
"""Simple timing scheduler."""
from collections import defaultdict
from typing import Callable, Dict, List

class Scheduler:
    def __init__(self):
        self.tasks: Dict[int, List[Callable]] = defaultdict(list)
        self.time = 0

    def every(self, interval: int, fn: Callable):
        self.tasks[interval].append(fn)

    def step(self):
        self.time += 1
        for interval, fns in self.tasks.items():
            if self.time % interval == 0:
                for fn in fns:
                    fn()
