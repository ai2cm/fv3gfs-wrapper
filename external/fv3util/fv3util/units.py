def ensure_equal_units(units1, units2):
    if not units_are_equal(units1, units2):
        raise UnitsError(f"incompatible units {units1} and {units2}")


def units_are_equal(units1, units2):
    return units1.strip() == units2.strip()


class UnitsError(Exception):
    pass
