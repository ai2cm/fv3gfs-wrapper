import unittest
import os
import shutil
from fv3gfs import (
    AliasDict, AliasRegistrationError, register_alias, register_fortran_aliases,
    dynamics_properties, physics_properties, get_tracer_metadata
)
from fv3gfs._alias_dict import reset_alias_dict_for_testing
import fv3config


base_dir = os.path.dirname(os.path.realpath(__file__))
run_dir = os.path.join(base_dir, 'rundir')


class AliasTests(unittest.TestCase):

    _original_directory = None

    def setUp(self):
        reset_alias_dict_for_testing()

    def tearDown(self):
        reset_alias_dict_for_testing()

    @classmethod
    def setUpClass(cls):
        config = fv3config.get_default_config()
        fv3config.write_run_directory(config, run_dir)
        cls._original_directory = os.getcwd()
        os.chdir(run_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._original_directory)
        shutil.rmtree(run_dir)

    def test_alias_dict_set(self):
        ad = AliasDict()
        ad['key'] = 'value'
        self.assertEqual(ad['key'], 'value')

    def test_alias_dict_initialize_from_dict(self):
        ad = AliasDict({'key1': 'value1', 'key2': 'value2'})
        self.assertEqual(ad['key1'], 'value1')
        self.assertEqual(ad['key2'], 'value2')
        self.assertEqual(len(ad), 2)

    def test_alias_dict_initialize_from_tuples(self):
        ad = AliasDict((('key1', 'value1'), ('key2', 'value2')))
        self.assertEqual(ad['key1'], 'value1')
        self.assertEqual(ad['key2'], 'value2')
        self.assertEqual(len(ad), 2)

    def test_register_alias(self):
        response = register_alias(key='value')
        self.assertEqual(response, None)

    def test_set_using_alias(self):
        register_alias(key='long_key')
        ad = AliasDict()
        ad['key'] = 'value'
        self.assertEqual(ad['long_key'], 'value')

    def test_get_using_alias(self):
        register_alias(key='long_key')
        ad = AliasDict()
        ad['long_key'] = 'value'
        self.assertEqual(ad['key'], 'value')

    def test_register_fortran_aliases_dynamics(self):
        register_fortran_aliases()
        ad = AliasDict()
        for properties in dynamics_properties:
            name = properties['name']
            with self.subTest(name):
                fortran_name = properties['fortran_name']
                value = f'{name}_value1'
                ad[fortran_name] = value
                self.assertEqual(ad[name], value)
                value = f'{name}_value2'
                ad[name] = value
                self.assertEqual(ad[fortran_name], value)

    def test_register_fortran_aliases_physics(self):
        register_fortran_aliases()
        ad = AliasDict()
        for properties in physics_properties:
            name = properties['name']
            with self.subTest(name):
                fortran_name = properties['fortran_name']
                value = f'{name}_value1'
                ad[fortran_name] = value
                self.assertEqual(ad[name], value)
                value = f'{name}_value2'
                ad[name] = value
                self.assertEqual(ad[fortran_name], value)

    def test_register_fortran_aliases_tracers(self):
        register_fortran_aliases()
        tracer_metadata = get_tracer_metadata()
        ad = AliasDict()
        for name, properties in tracer_metadata.items():
            fortran_name = properties['fortran_name']
            value = f'{name}_value1'
            ad[fortran_name] = value
            self.assertEqual(ad[name], value)
            value = f'{name}_value2'
            ad[name] = value
            self.assertEqual(ad[fortran_name], value)

    def test_tracers_exist(self):
        """We need there to be tracers so we can test whether tracer aliases are registered.
        """
        tracer_metadata = get_tracer_metadata()
        self.assertTrue(len(tracer_metadata) > 0)

    def test_cannot_register_after_initializing_alias_dict(self):
        ad = AliasDict()
        with self.assertRaises(AliasRegistrationError):
            register_alias(key='long_key')


if __name__ == '__main__':
    unittest.main()
