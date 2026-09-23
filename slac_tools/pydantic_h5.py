"""Generic pydantic-model <-> HDF5 serialization.

save_model(model, group, ...) writes every non-None field of a pydantic
model into an h5py.Group: nested BaseModels become subgroups, dicts
become subgroups keyed by dict key, lists/tuples/arrays become
datasets, and everything else (str/int/float/bool/datetime) becomes an
attribute.

load_model(model_cls, group, ...) is the mirror image: it rebuilds a
model_cls instance by reading each declared field back out of `group`
and calling model_cls(**fields).

Simple case, no overrides needed:

    class Measurement(pydantic.BaseModel):
        ...

    with h5py.File(path, "w") as f:
        save_model(m, f)
    with h5py.File(path, "r") as f:
        m = load_model(Measurement, f)

Optional overrides, shared by both functions unless noted. Together
they're the hooks for backward compatibility with older on-disk model
shapes: a model class can evolve (fields renamed, restructured, or
removed) while `load_model` still reads files written by a previous
version, by pointing these at the old field names/paths instead of
changing the generic dispatch itself.
- `name_map`: field name -> a different (optionally nested,
  "/"-separated) path in the file, e.g. {"fit_result": "analysis/fit_result"}.
- `manual`: field name -> a caller-supplied writer (save) or reader
  (load), used instead of the generic dispatch for a field whose
  on-disk shape doesn't fall out of its Python type. Also the escape hatch for
  storing/reading a custom on-disk data structure the generic dispatch
  has no case for at all (e.g. scan_ranges' dict[str, tuple[int, int]],
  flattened into paired *_start/*_end attributes instead of a subgroup).
- `skip`: field names to leave out of this call entirely.
- `extra` (load only): field name -> a value, or a callable given the
  already-resolved fields, supplied after the generic read -- e.g.
  reconstructed from another field that was just loaded. A field named
  in `extra` is implicitly skipped, so it never needs to also be named
  in `skip`.

`name_map`/`manual` propagate into nested recursive calls (they're
field-name -> behavior maps meant to apply wherever that field occurs);
`skip` does not, since a field name like "metadata" can mean something
different at each nesting depth.
"""

import collections.abc
import datetime
import h5py
import numpy
import pydantic
import types
import typing

ManualWriter = collections.abc.Callable[[typing.Any, h5py.Group], None]
ManualReader = collections.abc.Callable[[h5py.Group], typing.Any]

_NONE_AS_NAN = "none_as_nan"


def load_model(
    model_cls: type[pydantic.BaseModel],
    group: h5py.Group,
    name_map: dict[str, str] | None = None,
    manual: dict[str, ManualReader] | None = None,
    skip: set[str] | None = None,
    extra: dict[str, typing.Any] | None = None,
) -> pydantic.BaseModel:
    """Build a `model_cls` instance by reading its fields out of `group`.

    See module docstring for `name_map`/`manual`/`skip`/`extra`. `extra`
    values that are callables which compute a field from the fully resolved model.
    """
    name_map = name_map or {}
    manual = manual or {}
    skip = skip or set()
    extra = extra or {}
    resolved: dict[str, typing.Any] = {}
    for field_name, field in model_cls.model_fields.items():
        if field_name in skip or field_name in extra:
            continue
        if field_name in manual:
            value = manual[field_name](group)
        else:
            path = name_map.get(field_name, field_name)
            value = _read(group, path, field.annotation, name_map, manual)
        if value is not None:
            resolved[field_name] = value
    for field_name, value in extra.items():
        resolved[field_name] = value(resolved) if callable(value) else value
    return model_cls(**resolved)


def save_model(
    model: pydantic.BaseModel,
    group: h5py.Group,
    name_map: dict[str, str] | None = None,
    manual: dict[str, ManualWriter] | None = None,
    skip: set[str] | None = None,
) -> None:
    """Write every non-None field of `model` into `group`.

    `skip` only applies to this call's own fields, not to fields of the
    same name found while recursing into a nested model
    `name_map`/`manual` do propagate to nested models
    """
    name_map = name_map or {}
    manual = manual or {}
    skip = skip or set()
    for field_name in type(model).model_fields:
        if field_name in skip:
            continue
        value = getattr(model, field_name)
        if value is None:
            continue
        if field_name in manual:
            manual[field_name](value, group)
            continue
        _write(group, name_map.get(field_name, field_name), value, name_map, manual)


def _decode(value: typing.Any) -> typing.Any:
    return value.decode() if isinstance(value, bytes) else value


def _read(
    group: h5py.Group,
    path: str,
    annotation: typing.Any,
    name_map: dict[str, str],
    manual: dict[str, ManualReader],
) -> typing.Any:
    ann = _unwrap(annotation)
    origin = typing.get_origin(ann)

    if isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
        return _read_basemodel(group, path, ann, name_map, manual)
    if origin is dict:
        return _read_dict(group, path, ann, name_map, manual)
    if origin is list:
        return _read_list(group, path, ann, name_map, manual)
    if origin is tuple:
        return _read_tuple(group, path, ann)
    if ann is datetime.datetime:
        return _read_datetime(group, path)
    if ann is numpy.ndarray:
        return _read_ndarray(group, path)
    if ann is typing.Any:
        return _read_any(group, path)
    return _read_scalar(group, path, ann)


def _read_any(group: h5py.Group, path: str) -> typing.Any:
    # Read untyped values (e.g. `metadata: SerializeAsAny[Any]`)
    # holding an arbitrary dict or scalar value
    if path in group.attrs:
        return _decode(group.attrs[path])
    if path in group:
        return _read_leaf(group, path)
    return None


def _read_basemodel(
    group: h5py.Group,
    path: str,
    ann: type[pydantic.BaseModel],
    name_map: dict[str, str],
    manual: dict[str, ManualReader],
) -> pydantic.BaseModel | None:
    if path not in group:
        return None
    return load_model(ann, group[path], name_map, manual)


def _read_datetime(group: h5py.Group, path: str) -> datetime.datetime | None:
    if path not in group.attrs:
        return None
    return datetime.datetime.fromisoformat(group.attrs[path])


def _read_dict(
    group: h5py.Group,
    path: str,
    ann: typing.Any,
    name_map: dict[str, str],
    manual: dict[str, ManualReader],
) -> dict | None:
    """Read a `dict[K, V]` field. V is one declared type applied to every value"""
    if path not in group:
        return None
    sub = group[path]
    args = typing.get_args(ann)
    value_type = _unwrap(args[1]) if len(args) == 2 else typing.Any
    is_model = isinstance(value_type, type) and issubclass(
        value_type, pydantic.BaseModel
    )
    # scalar-valued keyed to attrs and non scalars to the group keys
    keys = set(sub.attrs.keys()).union(sub.keys())
    return {
        key: (
            load_model(value_type, sub[key], name_map, manual)
            if is_model
            else _read_leaf(sub, key)
        )
        for key in keys
    }


def _read_leaf(group: h5py.Group, key: str) -> typing.Any:
    """Reads container-of-Any fields (e.g. raw_data, metadata):
    plain attrs and Datasets are mapped directly, and
    a Group is recursively mapped into a dictionary."""
    if key in group.attrs:
        return _decode(group.attrs[key])
    item = group[key]
    if isinstance(item, h5py.Group):
        result = {k: _decode(v) for k, v in item.attrs.items()}
        result.update({k: _read_leaf(item, k) for k in item.keys()})
        return result
    return item[()]


def _read_list(
    group: h5py.Group,
    path: str,
    ann: typing.Any,
    name_map: dict[str, str],
    manual: dict[str, ManualReader],
) -> list | None:
    """Read a `list[X]` field. X is one declared type applied to every
    element (elements may be ragged in shape/length if X is ndarray, but
    not mixed types)"""
    args = typing.get_args(ann)
    elem_type = _unwrap(args[0]) if args else typing.Any
    elem_is_model = isinstance(elem_type, type) and issubclass(
        elem_type, pydantic.BaseModel
    )
    if elem_is_model or elem_type is numpy.ndarray:
        if path not in group:
            return None
        sub = group[path]
        if isinstance(sub, h5py.Dataset):
            # must be an empty dataset to reach here
            if sub.shape != (0,):
                raise AssertionError(
                    f"{path!r}: expected an empty dataset for an empty "
                    f"list[{elem_type.__name__}], got shape {sub.shape}"
                )
            return []
        indexes = sorted(int(i) for i in sub.keys())
        if elem_is_model:
            return [
                load_model(elem_type, sub[str(i)], name_map, manual) for i in indexes
            ]
        return [sub[str(i)][()] for i in indexes]
    if path in group.attrs:
        return [_decode(v) for v in group.attrs[path]]
    if path in group:
        return [_decode(v) for v in group[path][()]]
    return None


def _read_ndarray(group: h5py.Group, path: str) -> numpy.ndarray | None:
    if path not in group:
        return None
    ds = group[path]
    arr = ds[()]
    if ds.attrs.get(_NONE_AS_NAN):
        return numpy.array([None if numpy.isnan(v) else v for v in arr], dtype=object)
    return arr


def _read_scalar(group: h5py.Group, path: str, ann: typing.Any) -> typing.Any:
    # plain scalar: str / int / float / bool
    if path not in group.attrs:
        return None
    value = group.attrs[path]
    if ann in (str, int, float, bool):
        return ann(value) if ann is not str else _decode(value)
    return value


def _read_tuple(group: h5py.Group, path: str, ann: typing.Any) -> tuple | None:
    """Read a `tuple[X, ...]` field. Only the first declared type is used,
    applied to every element"""
    if path not in group.attrs and path not in group:
        return None
    elem_type = typing.get_args(ann)[0] if typing.get_args(ann) else float
    arr = group.attrs[path] if path in group.attrs else group[path][()]
    return tuple(elem_type(v) for v in arr)


def _unwrap(annotation: typing.Any) -> typing.Any:
    """Peel Optional[...] and Annotated[...] layers off a type annotation."""
    ann = annotation
    while True:
        if hasattr(ann, "__metadata__"):  # Annotated[X, ...]
            ann = typing.get_args(ann)[0]
            continue
        origin = typing.get_origin(ann)
        if origin is typing.Union or origin is types.UnionType:
            args = [a for a in typing.get_args(ann) if a is not type(None)]
            if len(args) == 1:
                ann = args[0]
                continue
        break
    return ann


def _write(
    group: h5py.Group,
    name: str,
    value: typing.Any,
    name_map: dict[str, str],
    manual: dict[str, ManualWriter],
) -> None:
    if isinstance(value, pydantic.BaseModel):
        save_model(value, group.create_group(name), name_map, manual)
    elif isinstance(value, dict):
        sub = group.create_group(name)
        for key, item in value.items():
            if item is None:
                continue
            _write(sub, key, item, name_map, manual)
    elif isinstance(value, datetime.datetime):
        group.attrs[name] = value.isoformat()
    elif isinstance(value, numpy.ndarray):
        if value.dtype == object:
            # None values can't be stored in hdf5 arrays
            # convert None to NaN with a none_as_nan tag for reading
            value = numpy.array(
                [numpy.nan if v is None else v for v in value], dtype=float
            )
            group.create_dataset(name, data=value).attrs[_NONE_AS_NAN] = True
        else:
            group.create_dataset(name, data=value)
    elif (
        isinstance(value, list)
        and value
        and isinstance(value[0], (pydantic.BaseModel, numpy.ndarray))
    ):
        # Store a list of BaseModel or ndarray (possibly ragged) as index-keyed entries
        sub = group.create_group(name)
        for i, elem in enumerate(value):
            _write(sub, str(i), elem, name_map, manual)
    elif isinstance(value, (list, tuple)):
        # Strings pass through as-is everything else is handed to numpy.asarray unchanged
        s = list(value)
        d = s if s and all(isinstance(v, str) for v in s) else numpy.asarray(s)
        group.create_dataset(name, data=d)
    elif isinstance(value, (str, int, float, bool, numpy.generic)):
        group.attrs[name] = value
    else:
        raise TypeError(
            f"field {name!r}: no generic dispatch for value of type "
            f"{type(value).__name__}; use `manual` for this field"
        )
