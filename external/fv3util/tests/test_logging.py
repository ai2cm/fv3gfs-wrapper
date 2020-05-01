from fv3util import capture_stream


def test_capture_stream():
    logging.basicConfig(level=logging.DEBUG)

    with capture_stream(sys.stdout):
        print("should appear")
        print("should appear")
        print("should appear")