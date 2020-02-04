
def apply_nudging(state, reference_state, nudging_timescales, timestep):
    """
    Nudge the given state towards the reference state according to the provided
    nudging timescales.

    Nudging is applied to the state in-place.

    Args:
        state (dict): A state dictionary.
        reference_state (dict): A reference state dictionary.
        nudging_timescales (dict): A dictionary whose keys are standard names and
            values are timedelta objects indicating the relaxation timescale for that
            variable.
        timestep (timedelta): length of the timestep

    Returns:
        nudging_tendencies (dict): A dictionary whose keys are standard names
            and values are DataArray objects indicating the nudging tendency
            of that standard name.
    """
    tendencies = get_nudging_tendencies(state, reference_state, nudging_timescales)
    _apply_tendencies(state, tendencies, timestep)
    return tendencies


def _apply_tendencies(state, tendencies, timestep):
    """Apply a dictionary of tendencies to a state, in-place. Assumes the tendencies
    are in units of <state units> per second.
    """
    for name, tendency in tendencies.items():
        if name not in state:
            raise ValueError(f'no state variable to apply tendency for {name}')
        state[name].values += tendency.values * timestep.total_seconds()


def get_nudging_tendencies(state, reference_state, nudging_timescales):
    """
    Return the nudging tendency of the given state towards the reference state
    according to the provided nudging timescales.

    Args:
        state (dict): A state dictionary.
        reference_state (dict): A reference state dictionary.
        nudging_timescales (dict): A dictionary whose keys are standard names and
            values are timedelta objects indicating the relaxation timescale for that
            variable.

    Returns:
        nudging_tendencies (dict): A dictionary whose keys are standard names
            and values are DataArray objects indicating the nudging tendency
            of that standard name.
    """
    return_array = {}
    for name, reference in reference_state.items():
        timescale = nudging_timescales[name].total_seconds()
        return_array[name] = (reference - state[name]) / timescale
        return_array[name].attrs['units'] = reference.attrs['units'] + ' s^-1'
    return return_array