import logging
import contextlib
import tempfile
import sys
import os


@contextlib.contextmanager
def capture_stream(stream, logger_name="fv3gfs"):

    # parent process:
    # close the reading end, we won't need this
    with tempfile.TemporaryFile() as out:
        try:
            orig_file_handle = os.dup(stream.fileno())
            # overwrite the streams fileno with a the pipe to be read by the forked
            # process below
            os.dup2(out.fileno(), stream.fileno())
            yield
        finally:
            # restore the original file handle
            os.dup2(orig_file_handle, stream.fileno())

            # print logging info
            logger = logging.getLogger(logger_name)
            out.seek(0)
            for line in out:
                logger.debug(line.strip().decode("UTF-8"))
