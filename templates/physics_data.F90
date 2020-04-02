module physics_data_mod

use atmosphere_mod, only: Atm, mytile
use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
use field_manager_mod,  only: MODEL_ATMOS
use atmos_model_mod, only: IPD_Data, IPD_Control, Atm_block
use dynamics_data_mod, only: i_start, i_end, j_start, j_end, nz
use iso_c_binding

implicit none

contains

    subroutine get_nz_soil_subroutine(nz_soil) bind(c)
        integer(c_int), intent(out) :: nz_soil
        nz_soil = IPD_Control%lsoil
    end subroutine get_nz_soil_subroutine

{% for item in physics_2d_properties %}
    subroutine set_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}({{ item.fortran_name }}) bind(c)
        real(c_double), intent(in), dimension(i_start():i_end(), j_start():j_end()) :: {{ item.fortran_name }}
        integer :: blocks_per_MPI_domain, i, j, i_block, i_column
        blocks_per_MPI_domain = Atm_block%nblks
        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do i_column = 1, Atm_block%blksz(i_block) ! points per block
                i = Atm_block%index(i_block)%ii(i_column)
                j = Atm_block%index(i_block)%jj(i_column)
                {% if "fortran_subname" in item %}
                    IPD_Data(i_block)%{{ item.container }}%{{ item.fortran_name }}(i_column)%{{ item.fortran_subname }} = {{ item.fortran_name }}(i, j)
                {% else %}
                    IPD_Data(i_block)%{{ item.container }}%{{ item.fortran_name }}(i_column) = {{ item.fortran_name }}(i, j)
                {% endif %}
            enddo
        enddo
    end subroutine set_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}

    subroutine get_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}({{ item.fortran_name }}_out) bind(c)
        real(c_double), intent(out), dimension(i_start():i_end(), j_start():j_end()) :: {{ item.fortran_name }}_out
        integer :: blocks_per_MPI_domain, i, j, i_block, i_column
        blocks_per_MPI_domain = Atm_block%nblks
        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do i_column = 1, Atm_block%blksz(i_block) ! points per block
                i = Atm_block%index(i_block)%ii(i_column)
                j = Atm_block%index(i_block)%jj(i_column)
                {% if "fortran_subname" in item %}
                    {{ item.fortran_name }}_out(i, j) = IPD_Data(i_block)%{{ item.container }}%{{ item.fortran_name }}(i_column)%{{ item.fortran_subname }}
                {% else %}
                    {{ item.fortran_name }}_out(i, j) = IPD_Data(i_block)%{{ item.container }}%{{ item.fortran_name }}(i_column)
                {% endif %}
            enddo
        enddo
    end subroutine get_{{ item.fortran_name }}{% if "fortran_subname" in item %}_{{ item.fortran_subname }}{% endif %}
{% endfor %}

{% for item in physics_3d_properties %}
    subroutine set_{{ item.fortran_name }}({{ item.fortran_name }}, nz_in) bind(c)
        integer, intent(in) :: nz_in
        real(c_double), intent(in), dimension(i_start():i_end(), j_start():j_end(), nz_in) :: {{ item.fortran_name }}
        integer :: blocks_per_MPI_domain, i, j, i_block, i_column, i_z
        blocks_per_MPI_domain = Atm_block%nblks
        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do i_column = 1, Atm_block%blksz(i_block) ! points per block
                i = Atm_block%index(i_block)%ii(i_column)
                j = Atm_block%index(i_block)%jj(i_column)
                do i_z = 1, nz_in
                    IPD_Data(i_block)%{{ item.container }}%{{ item.fortran_name }}(i_column, i_z) = {{ item.fortran_name }}(i, j, i_z)
                enddo
            enddo
        enddo
    end subroutine set_{{ item.fortran_name }}

    subroutine get_{{ item.fortran_name }}({{ item.fortran_name }}_out, nz_out) bind(c)
        integer, intent(in) :: nz_out
        real(c_double), intent(out), dimension(i_start():i_end(), j_start():j_end(), nz_out) :: {{ item.fortran_name }}_out
        integer :: blocks_per_MPI_domain, i, j, i_block, i_column, i_z
        blocks_per_MPI_domain = Atm_block%nblks
        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do i_column = 1, Atm_block%blksz(i_block) ! points per block
                i = Atm_block%index(i_block)%ii(i_column)
                j = Atm_block%index(i_block)%jj(i_column)
                do i_z = 1, nz_out
                    {{ item.fortran_name }}_out(i, j, i_z) = IPD_Data(i_block)%{{ item.container }}%{{ item.fortran_name }}(i_column, i_z)
                enddo
            enddo
        enddo
    end subroutine get_{{ item.fortran_name }}
{% endfor %}
end module physics_data_mod
