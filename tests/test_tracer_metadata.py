import unittest
import os
import shutil
from fv3gfs import get_tracer_metadata
import fv3config


base_dir = os.path.dirname(os.path.realpath(__file__))
run_dir = os.path.join(base_dir, 'rundir')

default_table = """# added by FRE: sphum must be present in atmos
# specific humidity for moist runs
 "TRACER", "atmos_mod", "sphum"
           "longname",     "specific humidity"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
# prognostic cloud water mixing ratio
 "TRACER", "atmos_mod", "liq_wat"
           "longname",     "cloud water mixing ratio"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
 "TRACER", "atmos_mod", "rainwat"
           "longname",     "rain mixing ratio"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
 "TRACER", "atmos_mod", "ice_wat"
           "longname",     "cloud ice mixing ratio"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
 "TRACER", "atmos_mod", "snowwat"
           "longname",     "snow mixing ratio"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
 "TRACER", "atmos_mod", "graupel"
           "longname",     "graupel mixing ratio"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
# prognostic ozone mixing ratio tracer
 "TRACER", "atmos_mod", "o3mr"
           "longname",     "ozone mixing ratio"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
# non-prognostic cloud amount
 "TRACER", "atmos_mod", "cld_amt"
           "longname",     "cloud amount"
           "units",        "1"
       "profile_type", "fixed", "surface_value=1.e30" /"""

default_metadata = {
    'specific_humidity': {
        'fortran_name': 'sphum',
        'units': 'kg/kg',
        'i_tracer': 1,
    },
    'cloud_water_mixing_ratio': {
        'fortran_name': 'liq_wat',
        'units': 'kg/kg',
        'i_tracer': 2,
    },
    'rain_mixing_ratio': {
        'fortran_name': 'rainwat',
        'units': 'kg/kg',
        'i_tracer': 3,
    },
    'cloud_ice_mixing_ratio': {
        'fortran_name': 'ice_wat',
        'units': 'kg/kg',
        'i_tracer': 4,
    },
    'snow_mixing_ratio': {
        'fortran_name': 'snowwat',
        'units': 'kg/kg',
        'i_tracer': 5,
    },
    'graupel_mixing_ratio': {
        'fortran_name': 'graupel',
        'units': 'kg/kg',
        'i_tracer': 6,
    },
    'ozone_mixing_ratio': {
        'fortran_name': 'o3mr',
        'units': 'kg/kg',
        'i_tracer': 7,
    },
    'cloud_amount': {
        'fortran_name': 'cld_amt',
        'units': '1',
        'i_tracer': 8,
    },
}

empty_table = ""
empty_metadata = {}

one_entry_table = """"TRACER", "atmos_mod", "sphum"
           "longname",     "specific humidity"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /"""
one_entry_metadata = {
    'specific_humidity': {
        'fortran_name': 'sphum',
        'units': 'kg/kg',
        'i_tracer': 1,
    },
}


class GetTracerMetadataTests(unittest.TestCase):

    maxDiff = 2000

    def tearDown(self):
        field_table_filename = os.path.join(os.getcwd(), 'field_table')
        if os.path.isfile(field_table_filename):
            os.remove(field_table_filename)

    def test_empty(self):
        with open('field_table', 'w') as file:
            file.write(empty_table)
        metadata = get_tracer_metadata()
        self.assertEqual(metadata, empty_metadata)

    def test_one_entry(self):
        with open('field_table', 'w') as file:
            file.write(one_entry_table)
        metadata = get_tracer_metadata()
        self.assertEqual(metadata, one_entry_metadata)

    def test_default(self):
        with open('field_table', 'w') as file:
            file.write(default_table)
        metadata = get_tracer_metadata()
        self.assertEqual(metadata, default_metadata)


if __name__ == '__main__':
    unittest.main()
