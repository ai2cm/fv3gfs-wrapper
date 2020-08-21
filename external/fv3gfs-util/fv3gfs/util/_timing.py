import contextlib
from typing import Mapping
from time import time


class Timer:
    def __init__(self):
        self._clock_starts = {}
        self._accumulated_time = {}

    def start(self, name):
        if name in self._clock_starts:
            raise ValueError(f"clock already started for '{name}'")
        else:
            self._clock_starts[name] = time()

    def stop(self, name):
        if name not in self._accumulated_time:
            self._accumulated_time[name] = time() - self._clock_starts.pop(name)
        else:
            self._accumulated_time[name] += time() - self._clock_starts.pop(name)

    @contextlib.contextmanager
    def clock(self, name):
        self.start(name)
        yield
        self.stop(name)

    @property
    def times(self) -> Mapping[str, float]:
        if len(self._clock_starts) > 0:
            raise RuntimeError(
                "Cannot retrieve times while clocks are still going: "
                f"{list(self._clock_starts.keys())}"
            )
        return self._accumulated_time.copy()
