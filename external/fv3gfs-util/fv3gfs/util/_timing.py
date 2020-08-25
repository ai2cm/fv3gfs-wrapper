import contextlib
from typing import Mapping
from time import time


class Timer:
    def __init__(self):
        self._clock_starts = {}
        self._accumulated_time = {}
        self._enabled = True

    def start(self, name):
        if self._enabled:
            if name in self._clock_starts:
                raise ValueError(f"clock already started for '{name}'")
            else:
                self._clock_starts[name] = time()

    def stop(self, name):
        if self._enabled:
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

    def reset(self):
        self._accumulated_time.clear()

    def enable(self):
        self._enabled = True

    def disable(self):
        if len(self._clock_starts) > 0:
            raise RuntimeError(
                "Cannot disable timer while clocks are still going: "
                f"{list(self._clock_starts.keys())}"
            )
        self._enabled = False

    @property
    def enabled(self):
        return self._enabled


class NullTimer(Timer):
    """A Timer class which does not actually accumulate timings.

    Meant to be used in place of an optional timer.
    """

    def start(self, name):
        pass

    def stop(self, name):
        pass
