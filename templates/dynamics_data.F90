module dynamics_data_mod

use atmosphere_mod, only: Atm, mytile
use tracers_mod
use iso_c_binding

implicit none

contains

{% for item in dynamics_properties %}
    subroutine set_{{ item.fortran_name }}({{ item.fortran_name }}_in) bind(c)
        real(c_double), intent(in), dimension({{ item.dim_ranges }}) :: {{ item.fortran_name }}_in
        Atm(mytile)%{{ item.fortran_name }}({{ item.dim_ranges }}) = {{ item.fortran_name }}_in({{ item.dim_ranges }})
    end subroutine set_{{ item.fortran_name }}

    subroutine get_{{ item.fortran_name }}({{ item.fortran_name }}_out) bind(c)
        real(c_double), intent(out), dimension({{ item.dim_ranges }}) :: {{ item.fortran_name }}_out
        {{ item.fortran_name }}_out({{ item.dim_ranges }}) = Atm(mytile)%{{ item.fortran_name }}({{ item.dim_ranges }})
    end subroutine get_{{ item.fortran_name }}
{% endfor %}

end module dynamics_data_mod
