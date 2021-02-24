"""Mainly for testing purposes.  This reproduces the results of run.py bit-for-bit."""
from ._wrapper import (
    initialize,
    get_step_count,
    step_dynamics,
    compute_radiation,
    compute_non_radiation_physics,
    apply_physics,
    save_intermediate_restart_if_enabled,
    cleanup,
)


if __name__ == "__main__":
    initialize()
    for i in range(get_step_count()):
        step_dynamics()
        compute_radiation()
        compute_non_radiation_physics()
        apply_physics()
        save_intermediate_restart_if_enabled()
    cleanup()
