import unittest
from fv3gfs import (
    AliasDict, AliasRegistrationError, register_alias, register_fortran_aliases,
    dynamics_properties, physics_properties, get_tracer_metadata
)
from fv3gfs._alias_dict import reset_alias_dict_for_testing


class AliasTests(unittest.TestCase):
    
    def setUp(self):
        reset_alias_dict_for_testing()

    def tearDown(self):
        reset_alias_dict_for_testing()

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
        for name, properties in dynamics_properties.items():
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
        for name, properties in physics_properties.items():
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

    def test_cannot_register_after_initializing_alias_dict(self):
        ad = AliasDict()
        with self.assertRaises(AliasRegistrationError):
            register_alias(key='long_key')


if __name__ == '__main__':
    unittest.main()
