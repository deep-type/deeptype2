from numpy import logical_and, logical_not, logical_or, ndarray, bool as np_bool


def logical_negate(truth, falses):
    out = truth
    assert isinstance(falses, (list, tuple)), "expected `falses` to be a list or tuple of numpy boolean arrays"
    for value in falses:
        assert isinstance(value, ndarray), "`falses` must be a list or tuple of numpy bool arrays but got: {}.".format(
            type(value))
        assert value.dtype == np_bool, "`falses` must be a list or tuple of numpy bool arrays but got dtype {}.".format(
            value.dtype)
        out = logical_and(out, logical_not(value))
    return out


def logical_ors(values):
    assert(len(values) > 0), "values cannot be empty."
    out = values[0]
    for val in values[1:]:
        out = logical_or(out, val)
    return out


def logical_ands(values):
    assert(len(values) > 0), "values cannot be empty."
    out = values[0]
    for val in values[1:]:
        out = logical_and(out, val)
    return out
