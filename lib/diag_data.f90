module physics_data_mod

    use atmosphere_mod, only: Atm, mytile
    use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
    use field_manager_mod,  only: MODEL_ATMOS
    use atmos_model_mod, only: IPD_Diag, Atm_block
    use dynamics_data_mod, only: i_start, i_end, j_start, j_end
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

    subroutine get_diagnostic_3d(idx, n, out) bind(c)
        integer(c_int), intent(in) :: idx, n
        real(c_double), intent(out), dimension(i_start():i_end(), j_start():j_end(), n) :: out
        ! locals
        integer :: blocks_per_MPI_domain, i, j, k, i_block, i_column, axes

        axes = IPD_Diag(idx)%axes

        blocks_per_MPI_domain = Atm_block%nblks
        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do i_column = 1, Atm_block%blksz(i_block) ! points per block
                i = Atm_block%index(i_block)%ii(i_column)
                j = Atm_block%index(i_block)%jj(i_column)
                if (axes == 3) then
                    do k = 1, n
                        out(i, j, k) = IPD_Diag(idx)%data(i_block)%var3(i_column, k)
                    end do
                else if (axes == 2) then
                    out(i, j, 1) = IPD_Diag(idx)%data(i_block)%var2(i_column)
                end if
            enddo
        enddo
    end subroutine

end module physics_data_mod

    