# -*- coding: utf-8 -*-
# cython: language_level=3, always_allow_keywords=True
# cython: c_string_type=unicode, c_string_encoding=utf8
cimport numpy as cnp
import numpy as np
import xarray as xr
from mpi4py import MPI
from ._exceptions import InvalidQuantityError
from datetime import datetime

ctypedef cnp.double_t REAL_t
real_type = np.float64

cdef extern:
    void initialize_subroutine(int *comm)
    void do_step_subroutine()
    void cleanup_subroutine()
    void do_dynamics()
    void do_physics()
    void save_intermediate_restart_if_enabled_subroutine()
    void save_intermediate_restart_subroutine(char *label, int *label_len)
    void initialize_time_subroutine(int *year, int *month, int *day, int *hour, int *minute, int *second)
    void get_time_subroutine(int *year, int *month, int *day, int *hour, int *minute, int *second)
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
    void set_tracer(int *i_tracer, REAL_t *array_in)
    void get_tracer_name(int *tracer_index, char *tracer_name_out, char *tracer_long_name_out, char *tracer_units_out)
    void get_num_cpld_calls(int *num_cpld_calls_out)
    void get_nz_soil_subroutine(int *nz_soil)
{% for item in physics_2d_properties %}
    void get_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_out)
    void set_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_in)
{% endfor %}
{% for item in physics_3d_properties %}
    void get_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_out, int *nz)
    void set_{{ item.fortran_name }}(REAL_t *{{ item.fortran_name }}_in, int *nz)
{% endfor %}



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


def get_dimension_lengths():
    cdef int nx, ny, nz, nz_soil
    get_centered_grid_dimensions(&nx, &ny, &nz)
    get_nz_soil_subroutine(&nz_soil)
    return {'nx': nx, 'ny': ny, 'nz': nz, 'nz_soil': nz_soil}


def get_array_from_dims(dim_name_list):
    cdef int nx, ny, nz, nz_soil
    get_centered_grid_dimensions(&nx, &ny, &nz)
    get_nz_soil_subroutine(&nz_soil)

    shape_list = []
    for dim_name in dim_name_list:
        if dim_name == 'x':
            shape_list.append(nx)
        elif dim_name == 'x_interface':
            shape_list.append(nx+1)
        elif dim_name == 'y':
            shape_list.append(ny)
        elif dim_name == 'y_interface':
            shape_list.append(ny+1)
        elif dim_name == 'z':
            shape_list.append(nz)
        elif dim_name == 'z_soil':
            shape_list.append(nz_soil)
        else:
            raise ValueError(f'{dim_name} is not a valid dimension name')
    return np.empty(shape_list, dtype=real_type)


def set_time(time):
    """Set model time to the given datetime.

    Does not change end time of the model run, or reset the step count.

    Args:
        time (datetime): the target time
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
    """Returns a datetime corresponding to the current model time.
    """
    cdef int year, month, day, hour, minute, second
    get_time_subroutine(&year, &month, &day, &hour, &minute, &second)
    return datetime(year, month, day, hour, minute, second)


def set_state(state):
    """
    Takes in a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are DataArrays containing that quantity's data. Sets the fortran state to
    those values.

    Arguments:
        state (dict): values to set
    """
    cdef REAL_t[:, :, ::1] input_value_3d
    cdef REAL_t[:, ::1] input_value_2d
    cdef REAL_t[::1] input_value_1d
    tracer_metadata = get_tracer_metadata()
    cdef set processed_names_set = set()
    for name, data_array in state.items():
        if name == 'time':
            set_time(state[name])
        elif len(data_array.shape) == 3:
            set_3d_quantity(name, data_array.values, data_array.shape[0], tracer_metadata)
        elif len(data_array.shape) == 2:
            set_2d_quantity(name, data_array.values)
        elif len(data_array.shape) == 1:
            set_1d_quantity(name, data_array.values)
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
        set_{{ item.fortran_name }}(&array[0, 0])
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


def get_state(names):
    """
    Returns a dictionary whose keys are quantity long names (with underscores instead of spaces)
    and values are DataArrays containing that quantity's data.

    Arguments:
        names (list of str, optional): A list of names to get.
    """
    cdef dict return_dict = {}
    cdef REAL_t[::1] array_1d
    cdef REAL_t[:, ::1] array_2d
    cdef REAL_t[:, :, ::1] array_3d
    cdef int nz, i_tracer
    cdef set input_names_set, processed_names_set
    input_names_set = set(names)

    if 'time' in input_names_set:
        return_dict['time'] = get_time()

{% for item in physics_2d_properties %}
    if '{{ item.name }}' in input_names_set:
        array_2d = get_array_from_dims({{ item.dims | safe }})
        get_{{ item.fortran_name }}(&array_2d[0, 0])
        return_dict['{{ item.name }}'] = xr.DataArray(
            np.asarray(array_2d),
            dims={{ item.dims | safe }},
            attrs={'units': '{{ item.units }}'}
        )
{% endfor %}

{% for item in physics_3d_properties %}
    if '{{ item.name }}' in input_names_set:
        array_3d = get_array_from_dims({{ item.dims | safe }})
        nz = array_3d.shape[0]
        get_{{ item.fortran_name }}(&array_3d[0, 0, 0], &nz)
        return_dict['{{ item.name }}'] = xr.DataArray(
            np.asarray(array_3d),
            dims={{ item.dims | safe }},
            attrs={'units': '{{ item.units }}'}
        )
{% endfor %}

{% for item in dynamics_properties %}
    {% if item.dims|length == 3 %}
    if '{{ item.name }}' in input_names_set:
        array_3d = get_array_from_dims({{ item.dims | safe }})
        get_{{ item.fortran_name }}(&array_3d[0, 0, 0])
        return_dict['{{ item.name }}'] = xr.DataArray(
            np.asarray(array_3d),
            dims={{ item.dims | safe }},
            attrs={'units': '{{ item.units }}'}
        )
    {% elif item.dims|length == 2 %}
    if '{{ item.name }}' in input_names_set:
        array_2d = get_array_from_dims({{ item.dims | safe }})
        get_{{ item.fortran_name }}(&array_2d[0, 0])
        return_dict['{{ item.name }}'] = xr.DataArray(
            np.asarray(array_2d),
            dims={{ item.dims | safe }},
            attrs={'units': '{{ item.units }}'}
        )
    {% elif item.dims|length == 1 %}
    if '{{ item.name }}' in input_names_set:
        array_1d = get_array_from_dims({{ item.dims | safe }})
        get_{{ item.fortran_name }}(&array_1d[0])
        return_dict['{{ item.name }}'] = xr.DataArray(
            np.asarray(array_1d),
            dims={{ item.dims | safe }},
            attrs={'units': '{{ item.units }}'}
        )
    {% endif %}
{% endfor %}

    for tracer_name, tracer_data in get_tracer_metadata().items():
        if (tracer_name in input_names_set):
            i_tracer = tracer_data['i_tracer']
            array_3d = get_array_from_dims(['z', 'y', 'x'])
            get_tracer(&i_tracer, &array_3d[0, 0, 0])
            return_dict[tracer_name] = xr.DataArray(
                np.asarray(array_3d),
                dims=['z', 'y', 'x'],
                attrs={'units': tracer_data['units']},
            )

    for name in names:
        if name not in return_dict:
            raise InvalidQuantityError(f'Quantity {name} does not exist - is there a typo?')
    return return_dict


cpdef dict get_tracer_metadata():
    """
    Returns a dict whose keys are tracer names and values are dictionaries with metadata.

    Metadata includes the keys 'i_tracer' (tracer index number in Fortran), 'fortran_name'
    (the short name in Fortran) and 'units'.
    """
    cdef dict out_dict = {}
    for i_tracer_minus_one, (tracer_name, tracer_long_name, tracer_units) in enumerate(get_tracer_metadata_list()):
        out_dict[str(tracer_long_name).replace(' ', '_')] = {
            'i_tracer': i_tracer_minus_one + 1,
            'fortran_name': tracer_name,
            'units': tracer_units
        }
    return out_dict

cdef list get_tracer_metadata_list():
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


def initialize():
    cdef int comm
    comm = MPI.COMM_WORLD.py2f()
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


def save_fortran_restart(label):
    cdef char* label_in = label
    cdef int label_len = len(label)
    save_intermediate_restart_subroutine(label_in, &label_len)


def cleanup():
    cleanup_subroutine()
