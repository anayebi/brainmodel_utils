import numpy as np


def dict_to_str(adict):
    """Converts a dictionary (e.g. hyperparameter configuration) into a string"""
    return "".join("{}{}".format(key, val) for key, val in sorted(adict.items()))


def check_np_equal(a, b):
    """Checks two numpy arrays are equal and works with nan values unlike np.array_equal.
    From: https://stackoverflow.com/questions/10710328/comparing-numpy-arrays-containing-nan"""
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def make_list(d, num_times):
    return [d] * num_times


def dict_app(d, curr):
    assert set(list(d.keys())) == set(list(curr.keys()))
    for k, v in curr.items():
        d[k].append(v)


def dict_np(d):
    for k, v in d.items():
        assert isinstance(v, list)
        d[k] = np.array(v)


def convert_to_list(inp):
    if not isinstance(inp, list):
        inp = [inp]
    return inp
