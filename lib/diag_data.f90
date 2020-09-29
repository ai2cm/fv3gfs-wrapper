module physics_data_mod

    use atmosphere_mod, only: Atm, mytile
    use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
    use field_manager_mod,  only: MODEL_ATMOS
    use atmos_model_mod, only: IPD_Diag, Atm_block
    use dynamics_data_mod, only: i_start, i_end, j_start, j_end, nz
    use iso_c_binding
    
    implicit none

    contains

    subroutine get_number_diagnostics(n) bind(c)
        integer(c_int), intent(out) :: n
        n = size(IPD_Diag)
    end subroutine

    subroutine get_metadata_diagnostics(idx, axes, mod_name, name, desc, unit) bind(c)
        integer(c_int), intent(in) :: idx
        integer(c_int), intent(out) :: axes
        character(c_char), intent(out) :: mod_name, name, desc, unit
        axes = IPD_Diag(idx)%axes
        mod_name = trim(IPD_Diag(idx)%mod_name) // c_null_char
        name = trim(IPD_Diag(idx)%name) // c_null_char
        desc = trim(IPD_Diag(idx)%desc) // c_null_char
        unit = trim(IPD_Diag(idx)%unit) // c_null_char
    end subroutine

    
end module physics_data_mod

    