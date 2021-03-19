module gfs_control_data_mod

    use atmos_model_mod, only: IPD_Control
    use iso_c_binding
    
    implicit none
    
    contains
        
    {% for item in gfs_control_properties %}
    subroutine get_{{ item.fortran_name }}({{ item.fortran_name }}_out) bind(c)
        {{ item.type_fortran }}({{ item.type_c }}), intent(out) :: {{ item.fortran_name }}_out
        {{ item.fortran_name }}_out = IPD_Control%{{ item.fortran_name }}
    end subroutine get_{{ item.fortran_name }}
    {% endfor %}
        
end module gfs_control_data_mod
    