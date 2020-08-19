# -*- coding: utf-8 -*-
# cython: language_level=3, always_allow_keywords=True
# cython: c_string_type=unicode, c_string_encoding=utf8
cimport numpy as cnp
import numpy as np
import fv3util
from mpi4py import MPI

ctypedef cnp.double_t REAL_t
real_type = np.float64
SURFACE_PRECIPITATION_RATE = 'surface_precipitation_rate'
MM_PER_M = 1000

cdef extern:
    void initialize_subroutine(int *comm)
    void do_step_subroutine()
    void cleanup_subroutine()
    void do_dynamics()
    void compute_physics_subroutine()
    void apply_physics_subroutine()
    void save_intermediate_restart_if_enabled_subroutine()
    void save_intermediate_restart_subroutine()
    void initialize_time_subroutine(int *year, int *month, int *day, int *hour, int *minute, int *second)
    void get_time_subroutine(int *year, int *month, int *day, int *hour, int *minute, int *second, int *fms_calendar_type)
    void get_physics_timestep_subroutine(int *physics_timestep)
    void get_centered_grid_dimensions(int *nx, int *ny, int *nz)
    void get_n_ghost_cells_subroutine(int *n_ghost)
    void get_u(REAL_t *u_out)
    void set_u(REAL_t *u_in)
    void get_v(REAL_t *v_out)
    void set_v(REAL_t *v_in)
{% for item in dynamics_properties %}
    void get_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_out)
    void set_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_in)
{% endfor %}
    void get_tracer_count(int *n_prognostic_tracers, int *n_total_tracers)
    void get_tracer(int *i_tracer, REAL_t *array_out)
    void get_tracer_breakdown(int *, int *, int *)
    void set_tracer(int *i_tracer, REAL_t *array_in)
    void get_tracer_name(int *tracer_index, char *tracer_name_out, char *tracer_long_name_out, char *tracer_units_out)
    void get_num_cpld_calls(int *num_cpld_calls_out)
    void get_nz_soil_subroutine(int *nz_soil)
{% for item in physics_2d_properties %}
    void get_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}(REAL_t *{{ item.fortran_name }}_out)
    void set_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}(REAL_t *{{ item.fortran_name }}_in)
{% endfor %}
{% for item in physics_3d_properties %}
    void get_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_out, int *nz)
    void set_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_in, int *nz)
{% endfor %}

cdef get_quantity_factory():
    cdef int nx, ny, nz, nz_soil
    get_centered_grid_dimensions(&nx, &ny, &nz)
    get_nz_soil_subroutine(&nz_soil)
    sizer = fv3util.SubtileGridSizer(
        nx,
        ny,
        nz,
        n_halo=fv3util.N_HALO_DEFAULT,
        extra_dim_lengths={
            fv3util.Z_SOIL_DIM: nz_soil,
        },
    )
    return fv3util.QuantityFactory(sizer, np)


cpdef int get_n_ghost_cells():
    """Return the number of ghost cells used by the Fortran dynamical core."""
    cdef int n_ghost
    get_n_ghost_cells_subroutine(&n_ghost)
    return n_ghost


def get_step_count():
    """Return the number of physics steps the Fortran model would like to complete
    before exiting, based on its configuration."""
    cdef int return_value
    get_num_cpld_calls(&return_value)
    return return_value


def get_dimension_lengths():
    """Return a dictionary specifying the (grid center) dimension lengths of the Fortran model."""
    cdef int nx, ny, nz, nz_soil
    get_centered_grid_dimensions(&nx, &ny, &nz)
    get_nz_soil_subroutine(&nz_soil)
    return {'nx': nx, 'ny': ny, 'nz': nz, 'nz_soil': nz_soil}


def set_time(time):
    """Set model time to the given datetime.

    Does not change end time of the model run, or reset the step count.

    Args:
        time (cftime.datetime or datetime.datetime): the target time
    """
    cdef int year, month, day, hour, minute, second
    year = time.year
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    second = time.second
    initialize_time_subroutine(&year, &month, &day, &hour, &minute, &second)


def get_time():
    """Returns a cftime.datetime corresponding to the current model time.
    """
    cdef int year, month, day, hour, minute, second, fms_calendar_type
    get_time_subroutine(&year, &month, &day, &hour, &minute, &second, &fms_calendar_type)
    return fv3util.FMS_TO_CFTIME_TYPE[fms_calendar_type](year, month, day, hour, minute, second)


def set_state(state):
    """
    Takes in a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are Quantity objects containing that quantity's data. Sets the fortran state to
    those values.

    Assumes quantity units are equivalent to what is used in the fortran code.

    Arguments:
        state (dict): values to set
    """
    cdef REAL_t[:, :, ::1] input_value_3d
    cdef REAL_t[:, ::1] input_value_2d
    cdef REAL_t[::1] input_value_1d
    tracer_metadata = get_tracer_metadata()
    cdef set processed_names_set = set()
    for name, quantity in state.items():
        if name == 'time':
            set_time(state[name])
        elif len(quantity.dims) == 3:
            set_3d_quantity(name, np.ascontiguousarray(quantity.view[:]), quantity.extent[0], tracer_metadata)
        elif len(quantity.dims) == 2:
            set_2d_quantity(name, np.ascontiguousarray(quantity.view[:]))
        elif len(quantity.dims) == 1:
            set_1d_quantity(name, np.ascontiguousarray(quantity.view[:]))
        else:
            raise ValueError(f'no setter available for {name}')


cdef int set_3d_quantity(name, REAL_t[:, :, ::1] array, int nz, dict tracer_metadata) except -1:
    cdef int i_tracer
    if False:
        pass  # need this so we can use elif in template
{% for item in dynamics_properties %}
    {% if item.dims|length == 3 %}
    elif name == '{{ item.name }}':
        set_{{ item.fortran_name }}(&array[0, 0, 0])
    {% endif %}
{% endfor %}
{% for item in physics_3d_properties %}
    elif name == '{{ item.name }}':
        set_{{ item.fortran_name }}(&array[0, 0, 0], &nz)
{% endfor %}
    elif name in tracer_metadata:
        i_tracer = tracer_metadata[name]['i_tracer']
        set_tracer(&i_tracer, &array[0, 0, 0])
    else:
        raise ValueError(f'no setter available for {name}')
    return 0


cdef int set_2d_quantity(name, REAL_t[:, ::1] array) except -1:
    if False:
        pass  # need this so we can use elif in template
{% for item in dynamics_properties %}
    {% if item.dims|length == 2 %}
    elif name == '{{ item.name }}':
        set_{{ item.fortran_name }}(&array[0, 0])
    {% endif %}
{% endfor %}
{% for item in physics_2d_properties %}
    elif name == '{{ item.name }}':
        set_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}(&array[0, 0])
{% endfor %}
    else:
        raise ValueError(f'no setter available for {name}')
    return 0


cdef int set_1d_quantity(name, REAL_t[::1] array) except -1:
    if False:
        pass  # need this so we can use elif in template
{% for item in dynamics_properties %}
    {% if item.dims|length == 1 %}
    elif name == '{{ item.name }}':
        set_{{ item.fortran_name }}(&array[0])
    {% endif %}
{% endfor %}
    else:
        raise ValueError(f'no setter available for {name}')
    return 0


def _get_quantity(state, name, allocator, dims, units, dtype):
    if name not in state:
        state[name] = allocator.empty(dims, units, dtype=dtype)
    return state[name]


def get_state(names, dict state=None, allocator=None):
    """
    Returns a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are Quantities containing that quantity's data.

    Does not copy halo values, regardless of whether the halo is allocated.

    Arguments:
        names (list of str, optional): A list of names to get.
        state (dict, optional): If given, update this state in-place with any retrieved
            quantities, and update any pre-existing quantities in-place with Fortran
            values.
        allocator (fv3util.QuantityFactory, optional): if given, use this to construct
            quantities. Otherwise use a QuantityFactory which uses the dimensions
            from the Fortran model with 3 allocated halo points.

    Returns:
        state (dict): state if given, otherwise a constructed state dictionary
    """
    if state is None:
        state = {}
    cdef REAL_t[::1] array_1d
    cdef REAL_t[:, ::1] array_2d
    cdef REAL_t[:, :, ::1] array_3d
    cdef int nz, i_tracer, dt_physics
    cdef set input_names_set, processed_names_set
    input_names_set = set(names)
    if allocator is None:
        allocator = get_quantity_factory()

    if 'time' in input_names_set:
        state['time'] = get_time()

{% for item in physics_2d_properties %}
    if '{{ item.name }}' in input_names_set:
        quantity = _get_quantity(state, "{{ item.name }}", allocator, {{ item.dims | safe }}, "{{ item.units }}", dtype=real_type)
        with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_2d:
            get_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}(&array_2d[0, 0])
{% endfor %}

{% for item in physics_3d_properties %}
    if '{{ item.name }}' in input_names_set:
        quantity = _get_quantity(state, "{{ item.name }}", allocator, {{ item.dims | safe }}, "{{ item.units }}", dtype=real_type)
        with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_3d:
            nz = array_3d.shape[0]
            get_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}(&array_3d[0, 0, 0], &nz)
{% endfor %}

{% for item in dynamics_properties %}
    {% if item.dims|length == 3 %}
    if '{{ item.name }}' in input_names_set:
        quantity = _get_quantity(state, "{{ item.name }}", allocator, {{ item.dims | safe }}, "{{ item.units }}", dtype=real_type)
        with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_3d:
            get_{{ item.fortran_name }}(&array_3d[0, 0, 0])
    {% elif item.dims|length == 2 %}
    if '{{ item.name }}' in input_names_set:
        quantity = _get_quantity(state, "{{ item.name }}", allocator, {{ item.dims | safe }}, "{{ item.units }}", dtype=real_type)
        with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_2d:
            get_{{ item.fortran_name }}(&array_2d[0, 0])
    {% elif item.dims|length == 1 %}
    if '{{ item.name }}' in input_names_set:
        quantity = _get_quantity(state, "{{ item.name }}", allocator, {{ item.dims | safe }}, "{{ item.units }}", dtype=real_type)
        with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_1d:
            get_{{ item.fortran_name }}(&array_1d[0])
    {% endif %}
{% endfor %}

    for tracer_name, tracer_data in get_tracer_metadata().items():
        i_tracer = tracer_data['i_tracer']
        if (tracer_name in input_names_set):
            quantity = _get_quantity(state, tracer_name, allocator, [fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_DIM], tracer_data["units"], dtype=real_type)
            with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_3d:
                get_tracer(&i_tracer, &array_3d[0, 0, 0])

    if SURFACE_PRECIPITATION_RATE in input_names_set:
        quantity = _get_quantity(state, SURFACE_PRECIPITATION_RATE, allocator, [fv3util.Y_DIM, fv3util.X_DIM], "mm/s", dtype=real_type)
        get_physics_timestep_subroutine(&dt_physics)
        with fv3util.recv_buffer(quantity.np.empty, quantity.view[:]) as array_2d:
            get_tprcp(&array_2d[0, 0])
        quantity.view[:] *= MM_PER_M / dt_physics

    for name in names:
        if name not in state:
            raise fv3util.InvalidQuantityError(
                f'Quantity {name} does not exist - is there a typo?'
            )
    return state


cpdef dict get_tracer_metadata():
    """
    Returns a dict whose keys are tracer names and values are dictionaries with metadata.

    Metadata includes the keys 'i_tracer' (tracer index number in Fortran), 'fortran_name'
    (the short name in Fortran), 'units', and a boolean 'is_water'.
    """
    cdef int n_prognostic_tracers, n_total_tracers, i_tracer
    # these lengths were chosen arbitrarily as "probably long enough"
    cdef char tracer_name[64]
    cdef char tracer_long_name[64]
    cdef char tracer_units[64]
    cdef int i

    cdef int n_water_tracers, dnats, pnats


    # get tracer counts
    get_tracer_count(&n_prognostic_tracers, &n_total_tracers)
    get_tracer_breakdown(&n_water_tracers, &dnats, &pnats)

    out_dict = {}
    for i_tracer in range(1, n_total_tracers + 1):

        get_tracer_name(&i_tracer, &tracer_name[0], &tracer_long_name[0], &tracer_units[0])
        is_water = i_tracer <= n_water_tracers
        fv3_python_name = str(tracer_long_name).replace(' ', '_')

        out_dict[fv3_python_name] = {
            'i_tracer': i_tracer,
            'fortran_name': tracer_name,
            'restart_name': tracer_name,
            'units': tracer_units,
            'is_water': is_water
        }


    return out_dict


def initialize():
    """Call initialization routines for the Fortran model."""
    cdef int comm
    comm = MPI.COMM_WORLD.py2f()
    initialize_subroutine(&comm)


def step():
    """Perform one dynamics-physics cycle of the Fortran model."""
    step_dynamics()
    step_physics()
    save_intermediate_restart_if_enabled_subroutine()


def step_dynamics():
    """Perform one physics step worth of dynamics in the Fortran model.

    Physics quantities are not updated by this routine."""
    do_dynamics()


def step_physics():
    """Perform a physics step in the Fortran model.

    Equivalent to calling compute_physics() and apply_physics() in that order."""
    compute_physics_subroutine()
    apply_physics_subroutine()


def compute_physics():
    """Call physics routines in the Fortran model and update physics prognostic state.

    It is necessary to call apply_physics() after this to update the dynamical
    prognostic state with the output from the routines called by this function."""
    compute_physics_subroutine()


def apply_physics():
    """Update dynamical prognostic state with output from physics routines."""
    apply_physics_subroutine()


def save_intermediate_restart_if_enabled():
    """If the Fortran model wants to do so on this timestep, write intermediate restart files.

    This function is used at the end of the Fortran main loop to replicate the
    intermediate restart behavior of the Fortran model.
    """
    save_intermediate_restart_if_enabled_subroutine()


def save_fortran_restart():
    """Trigger the Fortran model to write restart files."""
    save_intermediate_restart_subroutine()


def cleanup():
    """Call the Fortran cleanup routines, which clear memory and write final restart files."""
    cleanup_subroutine()
