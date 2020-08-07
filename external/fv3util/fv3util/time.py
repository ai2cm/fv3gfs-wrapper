import cftime


# Calendar constant values copied from time_manager in FMS
THIRTY_DAY_MONTHS = 1
JULIAN = 2
GREGORIAN = 3
NOLEAP = 4
FMS_TO_CFTIME_TYPE = {
    THIRTY_DAY_MONTHS: cftime.Datetime360Day,
    JULIAN: cftime.DatetimeJulian,
    GREGORIAN: cftime.DatetimeGregorian,  # Not a valid calendar in FV3GFS
    NOLEAP: cftime.DatetimeNoLeap,
}
