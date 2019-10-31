from ._exceptions import AliasRegistrationError

__all__ = ['register_alias', 'register_fortran_aliases', 'AliasDict']

_alias_to_long_name = {}

_alias_dict_created = False


def register_alias(**kwargs):
    """Register one or more aliases for long variable names.
    
    Usage:
        register_alias(T='air_temperature')
        my_aliases = {
            'q': 'specific_humidity'
        }
        register_alias(**my_aliases)
    """
    if _alias_dict_created:
        raise AliasRegistrationError(
            'Cannot register aliases after an AliasDict has been instantiated'
        )
    _alias_to_long_name.update(kwargs)


def register_fortran_aliases():
    """Register all fortran variable names as aliases for long names.
    """
    raise NotImplementedError()


def reset_alias_dict_for_testing():
    """This should only be used in testing code. Reset aliases and the flag
    of whether an AliasDict has been instantiated."""
    global _alias_dict_created
    _alias_to_long_name.clear()
    _alias_dict_created = False


class AliasDict(dict):

    def __init__(self, *args, **kwargs):
        global _alias_dict_created
        dict.__init__(self, *args, **kwargs)
        super(AliasDict, self).update(*args, **kwargs)
        _alias_dict_created = True

    def __getitem__(self, key):
        if key not in self and key in _alias_to_long_name:
            val = dict.__getitem__(self, _alias_to_long_name[key])
        else:
            val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        if key in _alias_to_long_name:
            key = _alias_to_long_name[key]
        dict.__setitem__(self, key, val)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).iteritems():
            self[k] = v
