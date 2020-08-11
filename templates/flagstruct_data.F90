module flagstruct_data_mod

use atmosphere_mod, only: Atm, mytile
use fv_nwp_nudge_mod, only: do_adiabatic_init
!use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
!use field_manager_mod,  only: MODEL_ATMOS
use iso_c_binding

implicit none

contains
    
{% for item in flagstruct_properties %}
    {% if item.fortran_name == "do_adiabatic_init" %}
    subroutine get_do_adiabatic_init(do_adiabatic_init_out) bind(c)
        logical(c_bool), intent(out) :: do_adiabatic_init_out
        do_adiabatic_init_out = do_adiabatic_init
    end subroutine get_do_adiabatic_init
    {% else %}
    subroutine get_{{ item.fortran_name }}({{ item.fortran_name }}_out) bind(c)
        {{ item.type_fortran }}({{ item.type_c }}), intent(out) :: {{ item.fortran_name }}_out
        {{ item.fortran_name }}_out = Atm(mytile)%flagstruct%{{ item.fortran_name }}
    end subroutine get_{{ item.fortran_name }}
    {% endif %}
{% endfor %}
    
end module flagstruct_data_mod
    