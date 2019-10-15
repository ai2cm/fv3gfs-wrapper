module dynamics_data_mod

use atmosphere_mod, only: Atm, mytile
use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
use field_manager_mod,  only: MODEL_ATMOS
use iso_c_binding

implicit none

contains

pure function i_start() result(i)
    integer :: i
    i = Atm(mytile)%bd%isd
end function i_start

pure function i_end() result(i)
    integer :: i
    i = Atm(mytile)%bd%ied
end function i_end

pure function j_start() result(j)
    integer :: j
    j = Atm(mytile)%bd%jsd
end function j_start

pure function j_end() result(j)
    integer :: j
    j = Atm(mytile)%bd%jed
end function j_end

pure function nz() result(n)
    integer :: n
    n = Atm(mytile)%flagstruct%npz
end function nz

subroutine get_centered_grid_dimensions(nx_out, ny_out, nz_out) bind(c)
    integer(c_int), intent(out) :: nx_out, ny_out, nz_out
    nx_out = i_end() - i_start() + 1 ! Fortran end index is inclusive
    ny_out = j_end() - j_start() + 1
    nz_out = nz()
end subroutine get_centered_grid_dimensions

subroutine get_n_ghost_cells_subroutine(n_ghost_out) bind(c)
    integer(c_int), intent(out) :: n_ghost_out
    n_ghost_out = Atm(mytile)%bd%ng
end subroutine get_n_ghost_cells_subroutine

subroutine get_u(u_out) bind(c)
    real(c_double), intent(out) :: u_out(i_start():i_end(), j_start():j_end()+1, nz())
    u_out(:, :, :) = Atm(mytile)%u(:, :, :)
end subroutine get_u

subroutine set_u(u_in) bind(c)
    real(c_double), intent(in) :: u_in(i_start():i_end(), j_start():j_end()+1, nz())
    Atm(mytile)%u(:, :, :) = u_in(:, :, :)
end subroutine set_u

subroutine get_v(v_out) bind(c)
    real(c_double), intent(out) :: v_out(i_start():i_end()+1, j_start():j_end(), nz())
    v_out(:, :, :) = Atm(mytile)%v(:, :, :)
end subroutine get_v

subroutine set_v(v_in) bind(c)
    real(c_double), intent(in) :: v_in(i_start():i_end()+1, j_start():j_end(), nz())
    Atm(mytile)%v(:, :, :) = v_in(:, :, :)
end subroutine set_v

subroutine get_temperature(T_out) bind(c)
    real(c_double), intent(out) :: T_out(i_start():i_end(), j_start():j_end(), nz())
    T_out(:, :, :) = Atm(mytile)%pt(:, :, :)
end subroutine get_temperature

subroutine set_temperature(T_in) bind(c)
    real(c_double), intent(in) :: T_in(i_start():i_end(), j_start():j_end(), nz())
    Atm(mytile)%pt(:, :, :) = T_in(:, :, :)
end subroutine set_temperature

subroutine get_delta_p(delta_p_out) bind(c)
    real(c_double), intent(out) :: delta_p_out(i_start():i_end(), j_start():j_end(), nz())
    delta_p_out(:, :, :) = Atm(mytile)%delp(:, :, :)
end subroutine get_delta_p

subroutine set_delta_p(delta_p_in) bind(c)
    real(c_double), intent(in) :: delta_p_in(i_start():i_end(), j_start():j_end(), nz())
    Atm(mytile)%delp(:, :, :) = delta_p_in(:, :, :)
end subroutine set_delta_p

subroutine get_prognostic_tracers(nq, q_out) bind(c)
    integer(c_int), intent(in) :: nq
    real(c_double), intent(out) :: q_out(i_start():i_end(), j_start():j_end(), nz(), nq)
    q_out(:, :, :, :) = Atm(mytile)%q(:, :, :, :)
end subroutine get_prognostic_tracers

subroutine set_prognostic_tracers(nq, q_in) bind(c)
    integer(c_int), intent(in) :: nq
    real(c_double), intent(out) :: q_in(i_start():i_end(), j_start():j_end(), nz(), nq)
    Atm(mytile)%q(:, :, :, :) = q_in(:, :, :, :)
end subroutine set_prognostic_tracers

subroutine get_diagnostic_tracers(nq_diag, q_diag_out) bind(c)
    integer(c_int), intent(in) :: nq_diag
    real(c_double), intent(out) :: q_diag_out(i_start():i_end(), j_start():j_end(), nz(), nq_diag)
    integer :: n_tracers, n_prog, i
    call get_number_tracers(MODEL_ATMOS, num_tracers=n_tracers, num_prog=n_prog)
    do i = n_prog+1, n_tracers
        q_diag_out(:, :, :, i - n_prog) = Atm(mytile)%qdiag(:, :, :, i)
    enddo
end subroutine get_diagnostic_tracers

subroutine set_diagnostic_tracers(nq_diag, q_diag_in) bind(c)
    integer(c_int), intent(in) :: nq_diag
    real(c_double), intent(in) :: q_diag_in(i_start():i_end(), j_start():j_end(), nz(), nq_diag)
    integer :: n_tracers, n_prog, i
    call get_number_tracers(MODEL_ATMOS, num_tracers=n_tracers, num_prog=n_prog)
    do i = n_prog+1, n_tracers
        Atm(mytile)%qdiag(:, :, :, i) = q_diag_in(:, :, :, i - n_prog)
    enddo
end subroutine set_diagnostic_tracers

subroutine get_tracer_count(n_prognostic_tracers, n_total_tracers) bind(c)
    integer(c_int), intent(out) :: n_prognostic_tracers, n_total_tracers
    call get_number_tracers(MODEL_ATMOS, num_tracers=n_total_tracers, num_prog=n_prognostic_tracers)
end subroutine get_tracer_count

subroutine get_tracer_name(tracer_index, tracer_name_out, tracer_long_name_out, tracer_units_out) bind(c)
    integer(c_int), intent(in) :: tracer_index
    character(kind=c_char), dimension(64) :: tracer_name_out, tracer_long_name_out, tracer_units_out
    character(kind=c_char, len=64) :: tracer_name, tracer_long_name, tracer_units
    integer i
    call get_tracer_names(model=MODEL_ATMOS, n=tracer_index, name=tracer_name, longname=tracer_long_name, units=tracer_units)
    tracer_name = trim(tracer_name) // c_null_char
    tracer_long_name = trim(tracer_long_name) // c_null_char
    tracer_units = trim(tracer_units) // c_null_char
    do i= 1, 64
        tracer_name_out(i) = tracer_name(i:i)
        tracer_long_name_out(i) = tracer_long_name(i:i)
        tracer_units_out(i) = tracer_units(i:i)
    enddo
end subroutine get_tracer_name

end module dynamics_data_mod
