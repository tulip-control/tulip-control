"""Auxiliary functions."""
import types
import typing as _ty


def n_tuple(
        length:
            int,
        item_type:
            type |
            None=None
        ) -> types.GenericAlias:
    """Return type of `length`-long tuples."""
    if item_type is None:
        item_type = _ty.Any
    item_types = length * (item_type,)
    generic = types.GenericAlias(
        tuple, item_types)
    # for static analysis with `pytype`
    return generic


def get_type(
        maybe_module:
            types.ModuleType |
            None,
        dotted_attr:
            str
        ) -> type:
    """Return requested type if present.

    If the module `maybe_module` has
    been imported (is not `None`),
    then return the requested type.

    Otherwise, return `typing.Any`.

    If the module *is* present,
    but the requested type:

    - does not exist:
      raise `AttributeError`

    - is not a `type`:
      raise `TypeError`
    """
    if maybe_module is None:
        return _ty.Any
    module = maybe_module
    path = dotted_attr.split('.')
    attr = module
    for step in path:
        attr = getattr(attr, step)
    if isinstance(attr, type):
        return attr
    raise TypeError(
        'Expected designated '
        'attribute to be a `type`, '
        'but instead found: '
        f'{type(attr) = }, '
        f'{attr = }')
