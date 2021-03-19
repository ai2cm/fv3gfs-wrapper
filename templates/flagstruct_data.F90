module flagstruct_data_mod

use atmosphere_mod, only: Atm, mytile
use atmos_model_mod, only: IPD_Control
use fv_nwp_nudge_mod, only: do_adiabatic_init
use iso_c_binding

implicit none

contains
    
{% for item in flagstruct_properties %}
    {% if item.fortran_name == "do_adiabatic_init" %}
    subroutine get_do_adiabatic_init(do_adiabatic_init_out) bind(c)
        logical(c_int), intent(out) :: do_adiabatic_init_out
        do_adiabatic_init_out = do_adiabatic_init
    end subroutine get_do_adiabatic_init
    {% else %}
    subroutine get_{{ item.fortran_name }}({{ item.fortran_name }}_out) bind(c)
        {{ item.type_fortran }}({{ item.type_c }}), intent(out) :: {{ item.fortran_name }}_out
        {% if item.location == "flagstruct" %}
        {{ item.fortran_name }}_out = Atm(mytile)%flagstruct%{{ item.fortran_name }}
        {% elif item.location == "Atm" %}
        {{ item.fortran_name }}_out = Atm(mytile)%{{ item.fortran_name }}
        {% elif item.location == "IPD_Control" %}
        {{ item.fortran_name }}_out = IPD_Control%{{ item.fortran_name }}
        {% endif %}
    end subroutine get_{{ item.fortran_name }}
    {% endif %}
{% endfor %}
    
end module flagstruct_data_mod
    
