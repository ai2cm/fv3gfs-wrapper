
! Created by  on 9/17/19.

module coupler_lib

!-----------------------------------------------------------------------
!
!   program that couples component models for the atmosphere,
!   ocean (amip), land, and sea-ice using the exchange module
!
!-----------------------------------------------------------------------

use time_manager_mod,  only: time_type, set_calendar_type, set_time,    &
                             set_date, days_in_month, month_name,       &
                             operator(+), operator (<), operator (>),   &
                             operator (/=), operator (/), operator (==),&
                             operator (*), THIRTY_DAY_MONTHS, JULIAN,   &
                             NOLEAP, NO_CALENDAR, date_to_string,       &
                             get_date
 
use  atmos_model_mod,  only: atmos_model_init, atmos_model_end,  &
                             update_atmos_model_dynamics,        &
                             update_atmos_radiation_physics,     &
                             update_atmos_model_state,           &
                             atmos_data_type, atmos_model_restart

use constants_mod,     only: constants_init
#ifdef INTERNAL_FILE_NML
use mpp_mod,            only: input_nml_file
#else
use fms_mod,            only: open_namelist_file
#endif
use       fms_mod,     only: file_exist, check_nml_error,               &
                             error_mesg, fms_init, fms_end, close_file, &
                             write_version_number, uppercase

use mpp_mod,           only: mpp_init, mpp_pe, mpp_root_pe, mpp_npes, mpp_get_current_pelist, &
                             mpp_set_current_pelist, stdlog, mpp_error, NOTE, FATAL, WARNING
use mpp_mod,           only: mpp_clock_id, mpp_clock_begin, mpp_clock_end, mpp_sync

use mpp_io_mod,        only: mpp_open, mpp_close, &
                             MPP_NATIVE, MPP_RDONLY, MPP_DELETE

use mpp_domains_mod,   only: mpp_get_global_domain, mpp_global_field, CORNER
use memutils_mod,      only: print_memuse_stats
use sat_vapor_pres_mod,only: sat_vapor_pres_init

use  diag_manager_mod, only: diag_manager_init, diag_manager_end, &
                             get_base_date, diag_manager_set_time_end

use data_override_mod, only: data_override_init

use iso_c_binding

implicit none

!-----------------------------------------------------------------------

character(len=128) :: version = '$Id: coupler_main.F90,v 19.0.4.1.2.3 2014/09/09 23:51:59 Rusty.Benson Exp $'
character(len=128) :: tag = '$Name: ulm_201505 $'

!-----------------------------------------------------------------------
!---- model defined-types ----

 type (atmos_data_type) :: Atm

!-----------------------------------------------------------------------
! ----- coupled model time -----

   type (time_type) :: Time_atmos, Time_init, Time_end,  &
                       Time_step_atmos, Time_step_ocean, &
                       Time_restart, Time_step_restart, Run_length
   integer :: num_cpld_calls, num_atmos_calls, nc, na, ret

! ----- coupled model initial date -----

   integer :: date_init(6)
   integer :: calendar_type = -99

! ----- timing flags -----

   integer :: initClock, mainClock, termClock
   integer, parameter :: timing_level = 1

! ----- namelist -----
   integer, dimension(6) :: current_date = (/ 0, 0, 0, 0, 0, 0 /)
   character(len=17) :: calendar = '                 '
   logical :: force_date_from_namelist = .false.  ! override restart values for date
   integer :: months=0, days=0, hours=0, minutes=0, seconds=0
   integer :: dt_atmos = 0
   integer :: dt_ocean = 0
   integer :: restart_days = 0
   integer :: restart_secs = 0
   integer :: atmos_nthreads = 1
   logical :: memuse_verbose = .false.
   logical :: use_hyper_thread = .false.
   logical :: debug_affinity = .false.
   integer :: ncores_per_node = 0

   namelist /coupler_nml/ current_date, calendar, force_date_from_namelist, &
                          months, days, hours, minutes, seconds,  &
                          dt_atmos, dt_ocean, atmos_nthreads, memuse_verbose, & 
                          use_hyper_thread, ncores_per_node, debug_affinity, &
                          restart_secs, restart_days

! ----- local variables -----
   character(len=32) :: timestamp
   logical :: intrm_rst

   public

contains

!#######################################################################

    subroutine get_num_cpld_calls(num_cpld_calls_out) bind(c)
      integer, intent(inout) :: num_cpld_calls_out
      num_cpld_calls_out = num_cpld_calls
    end subroutine get_num_cpld_calls

    subroutine module_init(comm) bind(c, name='initialize_subroutine')
        integer(c_int), intent(in) :: comm
        call mpp_init(localcomm=comm)
        call fms_init()
        initClock = mpp_clock_id( 'Initialization' )
        call mpp_clock_begin (initClock) !nesting problem
        call constants_init
        call sat_vapor_pres_init

        call coupler_init
        call print_memuse_stats('after coupler init')

        call mpp_set_current_pelist()
        call mpp_clock_end (initClock) !end initialization
        mainClock = mpp_clock_id( 'Main loop' )
        termClock = mpp_clock_id( 'Termination' )
        call mpp_clock_begin(mainClock) !begin main loop
        nc = 1
    end subroutine module_init

    subroutine do_dynamics() bind(c)
        Time_atmos = Time_atmos + Time_step_atmos
        call update_atmos_model_dynamics (Atm)
    end subroutine do_dynamics

    subroutine compute_physics_subroutine() bind(c)
        call update_atmos_radiation_physics (Atm)
    end subroutine compute_physics_subroutine

    subroutine apply_physics_subroutine() bind(c)
        call update_atmos_model_state (Atm)
    end subroutine apply_physics_subroutine

    subroutine save_intermediate_restart_if_enabled_subroutine() bind(c)
        if (intrm_rst) then
          if ((nc /= num_cpld_calls) .and. (Time_atmos == Time_restart)) then
            timestamp = date_to_string (Time_restart)
            call atmos_model_restart(Atm, timestamp)
            call coupler_res(timestamp)
            Time_restart = Time_restart + Time_step_restart
          endif
        endif
        nc = nc + 1
    end subroutine save_intermediate_restart_if_enabled_subroutine

    subroutine save_intermediate_restart_subroutine() bind(c)
        timestamp = date_to_string(Time_atmos)
        call atmos_model_restart(Atm, timestamp)
        call coupler_res(timestamp)
    end subroutine save_intermediate_restart_subroutine

    subroutine cleanup() bind(c, name='cleanup_subroutine')
#ifdef AVEC_TIMERS
        call avec_timers_output
#endif
        call mpp_set_current_pelist()
        call mpp_clock_end(mainClock)
        call mpp_clock_begin(termClock)

        call coupler_end
        call mpp_set_current_pelist()
        call mpp_clock_end(termClock)

        call fms_end
    end subroutine cleanup

    subroutine initialize_time(year, month, day, hour, minute, second) bind(c, name='initialize_time_subroutine')
        integer(c_int), intent(in) :: year, month, day, hour, minute, second
        integer :: date(6), date_init(6)

        date(1) = year
        date(2) = month
        date(3) = day
        date(4) = hour
        date(5) = minute
        date(6) = second

        !call diag_manager_init (TIME_INIT=date)

        call get_base_date ( date_init(1), date_init(2), date_init(3), &
                             date_init(4), date_init(5), date_init(6)  )

    !----- use current date if no base date ------

        if ( date_init(1) == 0 ) date_init = date

    !----- set initial and current time types ------

        Time_init  = set_date (date_init(1), date_init(2), date_init(3), &
                               date_init(4), date_init(5), date_init(6))

        Time_atmos = set_date (date(1), date(2), date(3),  &
                               date(4), date(5), date(6))

        num_atmos_calls = Time_step_ocean / Time_step_atmos
        Time_restart = Time_atmos + Time_step_restart

        Atm%Time_init = Time_init
        Atm%Time = Time_atmos
        Atm%Time_step = Time_step_atmos
        !call  atmos_model_init (Atm,  Time_init, Time_atmos, Time_step_atmos)
        end subroutine initialize_time

    subroutine get_time(year, month, day, hour, minute, second, fms_calendar_type) bind(c, name='get_time_subroutine')
        integer, intent(out) :: year, month, day, hour, minute, second, fms_calendar_type
        call get_date(Time_atmos, year, month, day, hour, minute, second)
        fms_calendar_type = calendar_type
    end subroutine get_time

    subroutine get_physics_timestep_subroutine(physics_timestep) bind(c)
        integer, intent(out) :: physics_timestep
        physics_timestep = dt_atmos
    end subroutine get_physics_timestep_subroutine


!#######################################################################

   subroutine coupler_init

!-----------------------------------------------------------------------
!   initialize all defined exchange grids and all boundary maps
!-----------------------------------------------------------------------
    integer :: total_days, total_seconds, unit, ierr, io
    integer :: n, gnlon, gnlat
    integer :: date(6), flags
    character(len=9) :: month
    logical :: use_namelist
    
    logical, allocatable, dimension(:,:) :: mask
    real,    allocatable, dimension(:,:) :: glon_bnd, glat_bnd
    integer :: omp_get_thread_num, get_cpu_affinity, base_cpu
!-----------------------------------------------------------------------
!----- initialization timing identifiers ----

!----- read namelist -------
!----- for backwards compatibilty read from file coupler.nml -----

#ifdef INTERNAL_FILE_NML
      read(input_nml_file, nml=coupler_nml, iostat=io)
      ierr = check_nml_error(io, 'coupler_nml')
#else
   if (file_exist('input.nml')) then
      unit = open_namelist_file ()
   else
      call error_mesg ('program coupler',  &
                       'namelist file input.nml does not exist', FATAL)
   endif
   
   ierr=1
   do while (ierr /= 0)
       read  (unit, nml=coupler_nml, iostat=io, end=10)
       ierr = check_nml_error (io, 'coupler_nml')
   enddo
10 call close_file (unit)
#endif

!----- write namelist to logfile -----
   call write_version_number (version, tag)
   if (mpp_pe() == mpp_root_pe()) write(stdlog(),nml=coupler_nml)

!----- allocate and set the pelist (to the global pelist) -----
    allocate( Atm%pelist  (mpp_npes()) )
    call mpp_get_current_pelist(Atm%pelist)

!----- read restart file -----

   if (file_exist('INPUT/coupler.res')) then
       call mpp_open( unit, 'INPUT/coupler.res', action=MPP_RDONLY )
       read (unit,*,err=999) calendar_type
       read (unit,*) date_init
       read (unit,*) date
       goto 998 !back to fortran-4
     ! read old-style coupler.res
   999 call mpp_close (unit)
       call mpp_open (unit, 'INPUT/coupler.res', action=MPP_RDONLY, form=MPP_NATIVE)
       read (unit) calendar_type
       read (unit) date
   998 call mpp_close(unit)
   else
       force_date_from_namelist = .true.
   endif       

!----- use namelist value (either no restart or override flag on) ---

 if ( force_date_from_namelist ) then

    if ( sum(current_date) <= 0 ) then
         call error_mesg ('program coupler',  &
              'no namelist value for current_date', FATAL)
    else
         date      = current_date
    endif

!----- override calendar type with namelist value -----

        select case( uppercase(trim(calendar)) )
        case( 'JULIAN' )
            calendar_type = JULIAN
        case( 'NOLEAP' )
            calendar_type = NOLEAP
        case( 'THIRTY_DAY' )
            calendar_type = THIRTY_DAY_MONTHS
        case( 'NO_CALENDAR' )
            calendar_type = NO_CALENDAR
        case default
            call mpp_error ( FATAL, 'COUPLER_MAIN: coupler_nml entry calendar must '// &
                                    'be one of JULIAN|NOLEAP|THIRTY_DAY|NO_CALENDAR.' )
        end select

 endif

!$      base_cpu = get_cpu_affinity()
!$      call omp_set_num_threads(atmos_nthreads)
!$OMP PARALLEL NUM_THREADS(atmos_nthreads)
!$      if(omp_get_thread_num() < atmos_nthreads/2 .OR. (.not. use_hyper_thread)) then  
!$         call set_cpu_affinity(base_cpu + omp_get_thread_num())
!$      else
!$         call set_cpu_affinity(base_cpu + omp_get_thread_num() + &
!$                               ncores_per_node - atmos_nthreads/2) 
!$      endif
!$      if (debug_affinity) then
!$        write(6,*) mpp_pe()," atmos  ",get_cpu_affinity(), base_cpu, omp_get_thread_num()
!$        call flush(6)
!$      endif
!$OMP END PARALLEL

    call set_calendar_type (calendar_type)

!----- write current/initial date actually used to logfile file -----

    if ( mpp_pe() == mpp_root_pe() ) then
      write (stdlog(),16) date(1),trim(month_name(date(2))),date(3:6)
    endif

 16 format ('  current date used = ',i4,1x,a,2i3,2(':',i2.2),' gmt') 

!-----------------------------------------------------------------------
!------ initialize diagnostics manager ------

    call diag_manager_init (TIME_INIT=date)

!----- always override initial/base date with diag_manager value -----

    call get_base_date ( date_init(1), date_init(2), date_init(3), &
                         date_init(4), date_init(5), date_init(6)  )

!----- use current date if no base date ------

    if ( date_init(1) == 0 ) date_init = date

!----- set initial and current time types ------

    Time_init  = set_date (date_init(1), date_init(2), date_init(3), &
                           date_init(4), date_init(5), date_init(6))

    Time_atmos = set_date (date(1), date(2), date(3),  &
                           date(4), date(5), date(6))

!-----------------------------------------------------------------------
!----- compute the ending time (compute days in each month first) -----
!
!   (NOTE: if run length in months then starting day must be <= 28)

    if ( months > 0 .and. date(3) > 28 )     &
        call error_mesg ('program coupler',  &
       'if run length in months then starting day must be <= 28', FATAL)

    Time_end = Time_atmos
    total_days = 0
    do n = 1, months
       total_days = total_days + days_in_month(Time_end)
       Time_end = Time_atmos + set_time (0,total_days)
    enddo

    total_days    = total_days + days
    total_seconds = hours*3600 + minutes*60 + seconds
    Run_length    = set_time (total_seconds,total_days)
    Time_end      = Time_atmos + Run_length

    !Need to pass Time_end into diag_manager for multiple thread case.
    call diag_manager_set_time_end(Time_end)


!-----------------------------------------------------------------------
!----- write time stamps (for start time and end time) ------

      call mpp_open( unit, 'time_stamp.out', nohdrs=.TRUE. )

      month = month_name(date(2))
      if ( mpp_pe() == mpp_root_pe() ) write (unit,20) date, month(1:3)

      call get_date (Time_end, date(1), date(2), date(3),  &
                               date(4), date(5), date(6))
      month = month_name(date(2))
      if ( mpp_pe() == mpp_root_pe() ) write (unit,20) date, month(1:3)

      call mpp_close (unit)

  20  format (6i4,2x,a3)

!-----------------------------------------------------------------------
!----- compute the time steps ------

Time_step_atmos = set_time (dt_atmos,0)
Time_step_ocean = set_time (dt_ocean,0)
num_cpld_calls  = Run_length / Time_step_ocean
num_atmos_calls = Time_step_ocean / Time_step_atmos
Time_step_restart = set_time (restart_secs, restart_days)
Time_restart = Time_atmos + Time_step_restart
intrm_rst = .false.
if (restart_days > 0 .or. restart_secs > 0) intrm_rst = .true.

!-----------------------------------------------------------------------
!------------------- some error checks ---------------------------------

!----- initial time cannot be greater than current time -------

    if ( Time_init > Time_atmos ) call error_mesg ('program coupler',  &
                    'initial time is greater than current time', FATAL)

!----- make sure run length is a multiple of ocean time step ------

    if ( num_cpld_calls * Time_step_ocean /= Run_length )  &
         call error_mesg ('program coupler',  &
         'run length must be multiple of ocean time step', FATAL)

! ---- make sure cpld time step is a multiple of atmos time step ----

    if ( num_atmos_calls * Time_step_atmos /= Time_step_ocean )  &
         call error_mesg ('program coupler',   &
         'atmos time step is not a multiple of the ocean time step', FATAL)

!------ initialize component models ------

      call  atmos_model_init (Atm,  Time_init, Time_atmos, Time_step_atmos)

      call print_memuse_stats('after atmos model init')

      call mpp_get_global_domain(Atm%Domain, xsize=gnlon, ysize=gnlat)
      allocate ( glon_bnd(gnlon+1,gnlat+1), glat_bnd(gnlon+1,gnlat+1) )
      call mpp_global_field(Atm%Domain, Atm%lon_bnd, glon_bnd, position=CORNER)
      call mpp_global_field(Atm%Domain, Atm%lat_bnd, glat_bnd, position=CORNER)

      call data_override_init ( ) ! Atm_domain_in  = Atm%domain, &
                                  ! Ice_domain_in  = Ice%domain, &
                                  ! Land_domain_in = Land%domain )

!-----------------------------------------------------------------------
!---- open and close dummy file in restart dir to check if dir exists --

      if (mpp_pe() == 0 ) then
         call mpp_open( unit, 'RESTART/file' )
         call mpp_close(unit, MPP_DELETE)
      endif

!-----------------------------------------------------------------------

   end subroutine coupler_init

!#######################################################################
   subroutine coupler_res(timestamp)
    character(len=32), intent(in) :: timestamp

    integer :: unit, date(6)

!----- compute current date ------

      call get_date (Time_atmos, date(1), date(2), date(3),  &
                                 date(4), date(5), date(6))

!----- write restart file ------

    if (mpp_pe() == mpp_root_pe())then
        call mpp_open( unit, 'RESTART/'//trim(timestamp)//'.coupler.res', nohdrs=.TRUE. )
        write( unit, '(i6,8x,a)' )calendar_type, &
             '(Calendar: no_calendar=0, thirty_day_months=1, julian=2, gregorian=3, noleap=4)'

        write( unit, '(6i6,8x,a)' )date_init, &
             'Model start time:   year, month, day, hour, minute, second'
        write( unit, '(6i6,8x,a)' )date, &
             'Current model time: year, month, day, hour, minute, second'
        call mpp_close(unit)
    endif
   end subroutine coupler_res

!#######################################################################

   subroutine coupler_end

   integer :: unit, date(6)
!-----------------------------------------------------------------------

      call atmos_model_end (Atm)

!----- compute current date ------

      call get_date (Time_atmos, date(1), date(2), date(3),  &
                                 date(4), date(5), date(6))

!----- check time versus expected ending time ----

      if (Time_atmos /= Time_end) call error_mesg ('program coupler',  &
              'final time does not match expected ending time', WARNING)

!----- write restart file ------

    call mpp_open( unit, 'RESTART/coupler.res', nohdrs=.TRUE. )
    if (mpp_pe() == mpp_root_pe())then
        write( unit, '(i6,8x,a)' )calendar_type, &
             '(Calendar: no_calendar=0, thirty_day_months=1, julian=2, gregorian=3, noleap=4)'

        write( unit, '(6i6,8x,a)' )date_init, &
             'Model start time:   year, month, day, hour, minute, second'
        write( unit, '(6i6,8x,a)' )date, &
             'Current model time: year, month, day, hour, minute, second'
    endif
    call mpp_close(unit)

!----- final output of diagnostic fields ----

   call diag_manager_end (Time_atmos)

!-----------------------------------------------------------------------
 
   end subroutine coupler_end

!#######################################################################

    subroutine get_diagnostics_count(n) bind(c)
        integer(c_int), intent(out) :: n
        n = size(Atm%Diag)
    end subroutine

    subroutine f_to_c_string(c, f)
        character(len=*) f
        character(kind=c_char, len=1), dimension(:) :: c
        ! local
        character(kind=c_char, len=128) trimmed
        integer i

        trimmed = trim(f) // c_null_char

        do i=1, size(c)
            c(i) = trimmed(i:i)
        end do
    end subroutine

    subroutine get_metadata_diagnostics(idx, axes, mod_name, name, desc, unit) bind(c)
        integer(c_int), intent(in) :: idx
        integer(c_int), intent(out) :: axes
        character(kind=c_char, len=1), dimension(128), intent(out) :: mod_name, name, desc, unit

        axes = Atm%Diag(idx)%axes
        call f_to_c_string(mod_name, Atm%Diag(idx)%mod_name) 
        call f_to_c_string(name, Atm%Diag(idx)%name)
        call f_to_c_string(desc, Atm%Diag(idx)%desc)
        call f_to_c_string(unit, Atm%Diag(idx)%unit)
    end subroutine

    subroutine get_diagnostic_3d(idx, out) bind(c)
        use dynamics_data_mod, only: i_start, i_end, j_start, j_end, nz
        use atmos_model_mod, only: Atm_block
        integer(c_int), intent(in) :: idx
        real(c_double), intent(out), dimension(i_start():i_end(), j_start():j_end(), nz()) :: out
        ! locals
        integer :: blocks_per_MPI_domain, i, j, k, i_block, i_column, axes, n
        n = nz()
        blocks_per_MPI_domain = Atm_block%nblks

        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do k=1, n
                do i_column = 1, Atm_block%blksz(i_block) ! points per block
                    i = Atm_block%index(i_block)%ii(i_column)
                    j = Atm_block%index(i_block)%jj(i_column)
                    out(i, j, n - k + 1) = Atm%Diag(idx)%data(i_block)%var3(i_column, k)
                end do
            enddo
        enddo
    end subroutine

    subroutine get_diagnostic_2d(idx, out) bind(c)
        use dynamics_data_mod, only: i_start, i_end, j_start, j_end
        use atmos_model_mod, only: Atm_block
        integer(c_int), intent(in) :: idx
        real(c_double), intent(out), dimension(i_start():i_end(), j_start():j_end()) :: out
        ! locals
        integer :: blocks_per_MPI_domain, i, j, k, i_block, i_column, axes

        blocks_per_MPI_domain = Atm_block%nblks
        do i_block = 1, blocks_per_MPI_domain ! blocks per MPI domain
            do i_column = 1, Atm_block%blksz(i_block) ! points per block
                i = Atm_block%index(i_block)%ii(i_column)
                j = Atm_block%index(i_block)%jj(i_column)
                out(i, j) = Atm%Diag(idx)%data(i_block)%var2(i_column)
            enddo
        enddo
    end subroutine


end module coupler_lib
