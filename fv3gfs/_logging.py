from fv3util import capture_stream


def _captured_stream(func):
    def myfunc(*args, **kwargs):
        with capture_stream(sys.stdout):
            return func(*args, **kwargs)

    return myfunc


def capture_fv3gfs_functions():
    """Surpress stderr and stdout from all fv3gfs functions
    
    The streams from this variables will be re-emited as `DEBUG` level logging 
    statements, which can be controlled using typical python `logging`.
    """
    import fv3gfs  # noqa

    for func in ["step_dynamics", "step_physics", "initialize", "cleanup"]:
        setattr(fv3gfs, func, captured_stream(getattr(fv3gfs, func)))