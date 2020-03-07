module update_dwinds_phys_mod

    use atmosphere_mod, only: Atm, mytile
    use fv_update_phys_mod, only: update_dwinds_phys

    implicit none

    contains

    pure function ic_start() result(ic)
        integer :: ic
        ic = Atm(mytile)%bd%isc
    end function ic_start

    pure function ic_end() result(ic)
        integer :: ic
        ic = Atm(mytile)%bd%iec
    end function ic_end

    pure function jc_start() result(jc)
        integer :: jc
        jc = Atm(mytile)%bd%jsc
    end function jc_start

    pure function jc_end() result(jc)
        integer :: jc
        jc = Atm(mytile)%bd%jec
    end function jc_end

    pure function id_start() result(id)
        integer :: id
        id = ic_start() - Atm(mytile)%bd%ng
    end function id_start

    pure function id_end() result(id)
        integer :: id
        id = ic_end() + Atm(mytile)%bd%ng
    end function id_end

    pure function jd_start() result(jd)
        integer :: jd
        jd = jc_start() - Atm(mytile)%bd%ng
    end function jd_start

    pure function jd_end() result(jd)
        integer :: jd
        jd = jc_end() + Atm(mytile)%bd%ng
    end function jd_end


    subroutine update_dwinds_phys_wrapped(dt, u_dt, v_dt, u, v) bind(c, name='update_dwinds_phys_subroutine')
        real,    intent(in) :: dt
        real, intent(inout) :: u(id_start():id_end(), jd_start():jd_end()+1, Atm(mytile)%npz)
        real, intent(inout) :: v(id_start():id_end()+1, jd_start():jd_end() ,Atm(mytile)%npz)
        real, intent(inout), dimension(id_start():id_end(), jd_start():jd_end(),Atm(mytile)%npz) :: u_dt, v_dt

        integer :: is, ie, js, je, isd, ied, jsd, jed, npx, npy, npz

        is = Atm(mytile)%bd%is
        ie = Atm(mytile)%bd%ie
        js = Atm(mytile)%bd%js
        je = Atm(mytile)%bd%je
        isd = id_start()
        ied = id_end()
        jsd = jd_start()
        jed = jd_end()
        npx = Atm(mytile)%npx
        npy = Atm(mytile)%npy
        npz = Atm(mytile)%npz

        call update_dwinds_phys(is, ie, js, je, isd, ied, jsd, jed, dt, u_dt, v_dt, u, v, &
                                Atm(mytile)%gridstruct, npx, npy, npz, Atm(mytile)%domain)

    end subroutine update_dwinds_phys_wrapped

end module update_dwinds_phys_mod
