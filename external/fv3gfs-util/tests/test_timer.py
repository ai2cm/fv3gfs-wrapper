from fv3gfs.util import Timer
import pytest
import time


@pytest.fixture
def timer():
    return Timer()


def test_start_stop(timer):
    timer.start("label")
    timer.stop("label")
    times = timer.times
    assert "label" in times
    assert len(times) == 1


def test_clock(timer):
    with timer.clock("label"):
        # small arbitrary computation task to time
        time.sleep(0.1)
    times = timer.times
    assert "label" in times
    assert len(times) == 1
    assert abs(times["label"] - 0.1) < 1e-2


def test_start_twice(timer):
    """cannot call start twice consecutively with no stop"""
    timer.start("label")
    with pytest.raises(ValueError) as err:
        timer.start("label")
    assert "clock already started for 'label'" in str(err.value)


def test_clock_in_clock(timer):
    """should not be able to create a given clock inside itself"""
    with timer.clock("label"):
        with pytest.raises(ValueError) as err:
            with timer.clock("label"):
                pass
    assert "clock already started for 'label'" in str(err.value)


def test_consecutive_start_stops(timer):
    """total time increases with consecutive clock blocks"""
    foo = 0
    timer.start("label")
    time.sleep(0.01)
    timer.stop("label")
    previous_time = timer.times["label"]
    for i in range(5):
        timer.start("label")
        time.sleep(0.01)
        timer.stop("label")
        assert timer.times["label"] > previous_time
        previous_time = timer.times["label"]


def test_consecutive_clocks(timer):
    """total time increases with consecutive clock blocks"""
    foo = 0
    with timer.clock("label"):
        time.sleep(0.01)
    previous_time = timer.times["label"]
    for i in range(5):
        with timer.clock("label"):
            time.sleep(0.01)
        assert timer.times["label"] > previous_time
        previous_time = timer.times["label"]