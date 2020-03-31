from ._wrapper import initialize, get_step_count, step, cleanup


if __name__ == "__main__":
    initialize()
    for i in range(get_step_count()):
        step()
    cleanup()
