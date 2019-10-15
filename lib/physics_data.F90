module physics_data_mod

use atmosphere_mod, only: Atm, mytile
use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
use field_manager_mod,  only: MODEL_ATMOS
use atmos_model_mod, only: IPD_Data, Atm_block
use dynamics_data_mod, only: i_start, i_end, j_start, j_end, nz
use iso_c_binding

implicit none

contains

subroutine get_physics_data(surface_sensible_heat_flux, surface_latent_heat_flux) bind(c)
    real(c_double), intent(out), dimension(i_start():i_end(), j_start():j_end()) :: surface_sensible_heat_flux, surface_latent_heat_flux
    integer :: blocks_per_MPI_domain, i, j, i_block, i_column
    blocks_per_MPI_domain = Atm_block%nblks
    do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
        do i_column = 1, Atm_block%blksz(i_block) ! points per block
            i = Atm_block%index(i_block)%ii(i_column)
            j = Atm_block%index(i_block)%jj(i_column)
            surface_latent_heat_flux(i, j) = IPD_Data(i_block)%Intdiag%dqsfc(i_column)
            surface_sensible_heat_flux(i, j) = IPD_Data(i_block)%Intdiag%dtsfc(i_column)
        enddo
    enddo
end subroutine get_physics_data


subroutine set_lhf(surface_latent_heat_flux) bind(c)
    real(c_double), intent(in), dimension(i_start():i_end(), j_start():j_end()) :: surface_latent_heat_flux
    integer :: blocks_per_MPI_domain, i, j, i_block, i_column
    blocks_per_MPI_domain = Atm_block%nblks
    do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
        do i_column = 1, Atm_block%blksz(i_block) ! points per block
            i = Atm_block%index(i_block)%ii(i_column)
            j = Atm_block%index(i_block)%jj(i_column)
            IPD_Data(i_block)%Intdiag%dqsfc(i_column) = surface_latent_heat_flux(i, j)
        enddo
    enddo
end subroutine set_lhf


subroutine set_shf(surface_sensible_heat_flux) bind(c)
    real(c_double), intent(in), dimension(i_start():i_end(), j_start():j_end()) :: surface_sensible_heat_flux
    integer :: blocks_per_MPI_domain, i, j, i_block, i_column
    blocks_per_MPI_domain = Atm_block%nblks
    do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
        do i_column = 1, Atm_block%blksz(i_block) ! points per block
            i = Atm_block%index(i_block)%ii(i_column)
            j = Atm_block%index(i_block)%jj(i_column)
            IPD_Data(i_block)%Intdiag%dtsfc(i_column) = surface_sensible_heat_flux(i, j)
        enddo
    enddo
end subroutine set_shf

end module physics_data_mod
