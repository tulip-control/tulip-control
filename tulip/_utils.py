"""Auxiliary functions."""
import types
import typing as _ty


def n_tuple(
        length:
            int,
        item_type:
            type |
            None=None
        ) -> type[tuple]:
    """Return type of `length`-long tuples."""
    if item_type is None:
        item_type = _ty.Any
    item_types = length * (item_type,)
    generic = types.GenericAlias(
        tuple, item_types)
    # for static analysis with `pytype`
    return generic
