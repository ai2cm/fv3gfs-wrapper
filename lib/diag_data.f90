module physics_data_mod

    use atmosphere_mod, only: Atm, mytile
    use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
    use field_manager_mod,  only: MODEL_ATMOS
    use atmos_model_mod, only: IPD_Diag, Atm_block
    use dynamics_data_mod, only: i_start, i_end, j_start, j_end, nz
    use iso_c_binding
    
    implicit none

    type, bind(c) :: external_diag_c_t
        integer(c_int) :: idx
        integer(c_int) :: axes
        character(c_char)    :: mod_name
        character(c_char)   :: name
        character(c_char)   :: desc
        character(c_char)    :: unit
    end type external_diag_c_t
    
    contains

    subroutine get_number_diagnostics(n)
        integer(c_int), intent(out) :: n
        n = size(IPD_Diag)
    end subroutine

    subroutine get_metadata_diagnostics(idx, metadata)
        integer(c_int), intent(in) :: idx
        type(external_diag_c_t), intent(out) :: metadata
        metadata%idx = idx
        metadata%axes = IPD_Diag(idx)%axes
        metadata%mod_name = trim(IPD_Diag(idx)%mod_name) // c_null_char
        metadata%name = trim(IPD_Diag(idx)%name) // c_null_char
        metadata%desc = trim(IPD_Diag(idx)%desc) // c_null_char
        metadata%unit = trim(IPD_Diag(idx)%desc) // c_null_char
    end subroutine

    
end module physics_data_mod

    