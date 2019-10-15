# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8
cimport numpy as cnp
import numpy as np
import xarray as xr

ctypedef cnp.double_t REAL_t
real_type = np.float64

cdef extern:
    void initialize_subroutine(int *comm)
    void do_step_subroutine()
    void cleanup_subroutine()
    void do_dynamics()
    void do_physics()
    void save_intermediate_restart_if_enabled_subroutine()
    void get_centered_grid_dimensions(int *nx, int *ny, int *nz)
    void get_n_ghost_cells_subroutine(int *n_ghost)
    void get_u(REAL_t *u_out)
    void set_u(REAL_t *u_in)
    void get_v(REAL_t *v_out)
    void set_v(REAL_t *v_in)
    void get_delta_p(REAL_t *delta_p_out)
    void set_delta_p(REAL_t *delta_p_in)
    void get_temperature(REAL_t *T_out)
    void set_temperature(REAL_t *T_in)
    void get_prognostic_tracers(int *nq, REAL_t *q_out)
    void set_prognostic_tracers(int *nq, REAL_t *q_in)
    void get_diagnostic_tracers(int *nq_diag, REAL_t *q_diag_out)
    void set_diagnostic_tracers(int *nq_diag, REAL_t *q_diag_in)
    void get_tracer_count(int *n_prognostic_tracers, int *n_total_tracers)
    void get_tracer_name(int *tracer_index, char *tracer_name_out, char *tracer_long_name_out, char *tracer_units_out)
    void get_physics_data(REAL_t *shf_out, REAL_t *lhf_out)
    void set_lhf(REAL_t *surface_latent_heat_flux)
    void set_shf(REAL_t *surface_sensible_heat_flux)
    void get_num_cpld_calls(int *num_cpld_calls_out)


def without_ghost_cells(state):
    cdef int n_ghost = get_n_ghost_cells()
    cdef int dimension_count
    state = state.copy()
    for name, value in state.items():
        dimension_count = len(value.shape)
        if dimension_count == 2:
            state[name] = value[n_ghost:-n_ghost, n_ghost:-n_ghost]
        elif dimension_count == 3:
            state[name] = value[:, n_ghost:-n_ghost, n_ghost:-n_ghost]
        elif dimension_count == 4:
            state[name] = value[:, :, n_ghost:-n_ghost, n_ghost:-n_ghost]
    return state


cpdef get_n_ghost_cells():
    cdef int n_ghost
    get_n_ghost_cells_subroutine(&n_ghost)
    return n_ghost


def get_step_count():
    cdef int return_value
    get_num_cpld_calls(&return_value)
    return return_value


def get_output_array(int nx_delta=0, int ny_delta=0, int nq=-1, bint include_z=True):
    cdef int nx, ny, nz
    cdef bint include_q_axis = nq != -1
    get_centered_grid_dimensions(&nx, &ny, &nz)
    assert not (include_q_axis and not include_z)  # tracers (q) should always be 3D
    if include_q_axis:
        shape = (nq, nz, ny + ny_delta, nx + nx_delta)
    elif include_z:
        shape = (nz, ny + ny_delta, nx + nx_delta)
    else:
        shape = (ny + ny_delta, nx + nx_delta)
    return np.empty(shape, dtype=real_type)


def set_state(dict state):
    cdef REAL_t[:, :, ::1] input_value_3d
    cdef REAL_t[:, ::1] input_value_2d
    for name, data_array in state.items():
        if len(data_array.shape) == 3:
            input_value_3d = data_array.values
            if name == 'air_temperature':
                set_temperature(&input_value_3d[0, 0, 0])
            elif name == 'eastward_wind':
                set_u(&input_value_3d[0, 0, 0])
            elif name == 'northward_wind':
                set_v(&input_value_3d[0, 0, 0])
            elif name == 'pressure_thickness_of_atmospheric_layer':
                set_delta_p(&input_value_3d[0, 0, 0])
        elif len(data_array.shape) == 2:
            input_value_2d = data_array.values
            if name == 'surface_latent_heat_flux':
                set_lhf(&input_value_2d[0, 0])
            if name == 'surface_sensible_heat_flux':
                set_shf(&input_value_2d[0, 0])
    set_tracer_state(state)


cdef void set_tracer_state(dict state):
    cdef REAL_t[:, :, :, ::1] q_prog
    cdef REAL_t[:, :, :, ::1] q_diag
    cdef REAL_t[:, :, ::1] input_array
    cdef REAL_t[:, :, ::1] target_array
    cdef list tracer_list = get_tracer_metadata()
    cdef str short_name, long_name, units
    cdef int n_prognostic_tracers, n_total_tracers
    cdef int i_name, i, j, k
    cdef int nx, ny, nz
    get_tracer_count(&n_prognostic_tracers, &n_total_tracers)
    cdef int n_diagnostic_tracers = n_total_tracers - n_prognostic_tracers
    cdef bint modified_prognostic = False
    cdef bint modified_diagnostic = False
    get_centered_grid_dimensions(&nx, &ny, &nz)
    # Must check first if prognostic/diagnostic is actually modified before we
    # copy out an entire array for no reason
    i_name = 0
    for short_name, long_name, units in tracer_list:
        state_name = long_name.replace(' ', '_')
        if state_name in state:
            if i_name < n_prognostic_tracers:
                modified_prognostic = True
            else:
                modified_diagnostic = True
        i_name += 1
    if modified_prognostic:  # only get the array if it's modified
        q_prog = get_output_array(nq=n_prognostic_tracers)
        get_prognostic_tracers(&n_prognostic_tracers, &q_prog[0, 0, 0, 0])
    if modified_diagnostic:
        q_diag = get_output_array(nq=n_total_tracers - n_prognostic_tracers)
        get_diagnostic_tracers(&n_diagnostic_tracers, &q_diag[0, 0, 0, 0])
    # Same loop as before, but now we actually modify the values
    i_name = 0
    for short_name, long_name, units in tracer_list:
        state_name = long_name.replace(' ', '_')
        if state_name in state:
            input_array = state[state_name].values
            if i_name < n_prognostic_tracers:
                target_array = q_prog[i_name, :, :, :]
            else:
                i_diag = i - n_prognostic_tracers
                target_array = q_diag[i_diag, :, :, :]
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        target_array[k, j, i] = input_array[k, j, i]
        i_name += 1
    # The values aren't the same memory as the Fortran data, we have to copy it back in
    if modified_prognostic:
        set_prognostic_tracers(&n_prognostic_tracers, &q_prog[0, 0, 0, 0])
    if modified_diagnostic:
        set_diagnostic_tracers(&n_diagnostic_tracers, &q_diag[0, 0, 0, 0])


def get_state():
    """
    Returns a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are DataArrays containing that quantity's data. Includes ghost cells.
    """
    cdef dict return_dict = {}
    return_dict.update(get_physics_state())
    return_dict.update(get_tracer_state())
    return_dict.update(get_dynamics_state())
    return return_dict


cpdef dict get_physics_state():
    """
    Returns a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are DataArrays containing that quantity's data. Includes ghost cells.
    """
    cdef REAL_t[:, ::1] lhf = get_output_array(include_z=False)
    cdef REAL_t[:, ::1] shf = get_output_array(include_z=False)
    get_physics_data(&shf[0, 0], &lhf[0, 0])
    return {
        'surface_latent_heat_flux': xr.DataArray(
            np.asarray(lhf),
            dims=['y', 'x'],
            attrs={'units': 'W/m^2'}
        ),
        'surface_sensible_heat_flux': xr.DataArray(
            np.asarray(shf),
            dims=['y', 'x'],
            attrs={'units': 'W/m^2'}
        ),
    }


cpdef dict get_dynamics_state():
    """
    Returns a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are DataArrays containing that quantity's data. Includes ghost cells.
    """
    cdef REAL_t[:, :, ::1] u = get_output_array(ny_delta=1)
    cdef REAL_t[:, :, ::1] v = get_output_array(nx_delta=1)
    cdef REAL_t[:, :, ::1] delta_p = get_output_array()
    cdef REAL_t[:, :, ::1] T = get_output_array()
    get_u(&u[0, 0, 0])
    get_v(&v[0, 0, 0])
    get_delta_p(&delta_p[0, 0, 0])
    get_temperature(&T[0, 0, 0])
    return {
        'eastward_wind': xr.DataArray(
            np.asarray(u),
            dims=['z', 'y_interface', 'x'],
            attrs={'units': 'm/s'},
        ),
        'northward_wind': xr.DataArray(
            np.asarray(v),
            dims=['z', 'y', 'x_interface'],
            attrs={'units': 'm/s'},
        ),
        'air_temperature': xr.DataArray(
            np.asarray(T),
            dims=['z', 'y', 'x'],
            attrs={'units': 'm/s'}
        ),
        'pressure_thickness_of_atmospheric_layer': xr.DataArray(
            np.asarray(delta_p),
            dims=['z', 'y', 'x'],
            attrs={'units': 'm/s'},
        ),
    }


cpdef dict get_tracer_state():
    """
    Returns a dictionary whose keys are tracer long names (with underscores instead of spaces)
    and values are DataArrays containing that tracer's data. Includes ghost cells.
    """
    cdef int n_prognostic_tracers, n_total_tracers, i
    cdef REAL_t[:, :, :, ::1] q_prog
    cdef REAL_t[:, :, :, ::1] q_diag
    cdef list tracer_name_list = get_tracer_metadata()
    cdef str name
    get_tracer_count(&n_prognostic_tracers, &n_total_tracers)
    cdef int n_diagnostic_tracers = n_total_tracers - n_prognostic_tracers
    if n_prognostic_tracers > 0:
        q_prog = get_output_array(nq=n_prognostic_tracers)
        get_prognostic_tracers(&n_prognostic_tracers, &q_prog[0, 0, 0, 0])
    if n_prognostic_tracers < n_total_tracers:
        q_diag = get_output_array(nq=n_total_tracers - n_prognostic_tracers)
        get_diagnostic_tracers(&n_diagnostic_tracers, &q_diag[0, 0, 0, 0])
    cdef dict out_dict = {}
    for i in range(n_total_tracers):
        name = tracer_name_list[i][1].replace(' ', '_')
        if i < n_prognostic_tracers:
            out_dict[name] = xr.DataArray(
                np.asarray(q_prog[i, :, :, :]),
                dims=['z', 'y', 'x'],
                attrs={'units': tracer_name_list[i][2], 'short_name': tracer_name_list[i][0]},
            )
        else:
            out_dict[name] = xr.DataArray(
                np.asarray(q_diag[i - n_prognostic_tracers, :, :, :]),
                dims=['z', 'y', 'x'],
                attrs={'units': tracer_name_list[i][2], 'short_name': tracer_name_list[i][0]},
            )
    return out_dict


cdef list get_tracer_metadata():
    """
    Returns a list [(tracer_name, tracer_long_name, tracer_units), (...), ...] of tracer short
    names, long names, and units in the index order of the Fortran tracer code.
    """
    cdef list out_list = []
    cdef int n_prognostic_tracers, n_total_tracers, i_tracer
    # these lengths were chosen arbitrarily as "probably long enough"
    cdef char tracer_name[64]
    cdef char tracer_long_name[64]
    cdef char tracer_units[64]
    cdef int i
    get_tracer_count(&n_prognostic_tracers, &n_total_tracers)
    for i_tracer in range(1, n_total_tracers + 1):
        get_tracer_name(&i_tracer, &tracer_name[0], &tracer_long_name[0], &tracer_units[0])
        out_list.append((tracer_name, tracer_long_name, tracer_units))
    return out_list


def initialize(int comm):
    initialize_subroutine(&comm)


def step():
    do_dynamics()
    do_physics()
    save_intermediate_restart_if_enabled_subroutine()


def step_dynamics():
    do_dynamics()


def step_physics():
    do_physics()


def save_intermediate_restart_if_enabled():
    save_intermediate_restart_if_enabled_subroutine()


def cleanup():
    cleanup_subroutine()
