module flagstruct_data_mod

use atmosphere_mod, only: Atm, mytile
use fv_nwp_nudge_mod, only: do_adiabatic_init
!use tracer_manager_mod, only: get_tracer_names, get_number_tracers, get_tracer_index
!use field_manager_mod,  only: MODEL_ATMOS
use iso_c_binding

implicit none

contains
    

    
    subroutine get_n_split(n_split_out) bind(c)
        integer(c_int), intent(out) :: n_split_out
        
        n_split_out = Atm(mytile)%flagstruct%n_split
        
    end subroutine get_n_split
    

    
    subroutine get_consv_te(consv_te_out) bind(c)
        real(c_double), intent(out) :: consv_te_out
        
        consv_te_out = Atm(mytile)%flagstruct%consv_te
        
    end subroutine get_consv_te
    

    
    subroutine get_ks(ks_out) bind(c)
        integer(c_int), intent(out) :: ks_out
        
        ks_out = Atm(mytile)%ks
        
    end subroutine get_ks
    

    
    subroutine get_ptop(ptop_out) bind(c)
        real(c_double), intent(out) :: ptop_out
        
        ptop_out = Atm(mytile)%ptop
        
    end subroutine get_ptop
    

    
    subroutine get_do_adiabatic_init(do_adiabatic_init_out) bind(c)
        logical(c_bool), intent(out) :: do_adiabatic_init_out
        do_adiabatic_init_out = do_adiabatic_init
    end subroutine get_do_adiabatic_init
    

    
end module flagstruct_data_mod
    