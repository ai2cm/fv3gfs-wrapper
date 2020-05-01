import sys
from unittest.mock import Mock
import pytest
import os
import ctypes

from fv3util import capture_stream


def get_libc():
    if os.uname().sysname == 'Linux':
        return ctypes.cdll.LoadLibrary("libc.so.6")
    else:
        pytest.skip()


def test_capture_stream_python_print():
    logger =  Mock()
    text = "hello world"

    with capture_stream(sys.stdout, logger=logger):
        print(text)

    logger.debug.assert_called_with(text)