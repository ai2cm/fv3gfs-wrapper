import contextlib
from .utils import is_contiguous

BUFFER_CACHE = {}


@contextlib.contextmanager
def array_buffer(allocator, shape, dtype):
    """
    Returns a communications buffer using an allocator.
    Buffers will be re-used between subsequent calls.

    tag is a dummy argument to manage returning distinct cached buffers
    """
    key = (allocator, shape, dtype)
    if key in BUFFER_CACHE and len(BUFFER_CACHE[key]) > 0:
        array = BUFFER_CACHE[key].pop()
        print("reusing array", array.data)
        yield array
    else:
        if key not in BUFFER_CACHE:
            BUFFER_CACHE[key] = []
        array = allocator(shape, dtype=dtype)
        print("allocated array", array.data)
        yield array
    print("array released", array.data)
    BUFFER_CACHE[key].append(array)


@contextlib.contextmanager
def send_buffer(numpy, array):
    if array is None or is_contiguous(array):
        yield array
    else:
        with array_buffer(numpy.empty, array.shape, array.dtype) as sendbuf:
            sendbuf[:] = array
            yield sendbuf


@contextlib.contextmanager
def recv_buffer(numpy, array):
    if array is None or is_contiguous(array):
        yield array
    else:
        with array_buffer(numpy.empty, array.shape, array.dtype) as recvbuf:
            yield recvbuf
            array[:] = recvbuf
