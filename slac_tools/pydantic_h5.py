"""Generic pydantic-model <-> HDF5 serialization.

`save_model` walks a pydantic BaseModel's fields and writes them
into an h5py.Group: nested BaseModels become subgroups, dicts become
subgroups keyed by dict key (recursively, so a dict of dicts of arrays
works without special casing), arrays/lists/tuples become datasets, and
everything else (str/int/float/bool/datetime) becomes an attribute. A
single `is not None` check decides whether a field is written at all.
An object-dtype ndarray (e.g. a tuple like `(1.25, None)` turned into an
array by a field's validator) is written with any `None` entries
converted to NaN, since h5py can't store an object-dtype array at all --
and the dataset is tagged with a `none_as_nan` attr so `load_model` knows
to convert those NaNs back to None on read. A plain float array's NaN
(never object-dtype to begin with, so never tagged) reads back as NaN
unchanged -- the tag, not a NaN/Optional heuristic, is what disambiguates
the two, since a written-out NaN is byte-identical either way.

`load_model` is the mirror image: it walks the *model class's*
declared field types (via pydantic's `model_fields[name].annotation`)
rather than a live value's runtime type, since there's nothing else to
dispatch on when reading raw h5py attrs/datasets back. Container-of-Any
fields (like `raw_data: dict[str, Any]`) fall back to inspecting the
h5py node itself (Group vs Dataset), the same way the saver did.

A `list[SomeModel]` field (elements are BaseModel instances, not plain
scalars/arrays) gets one subgroup per element, named by its index
("0", "1", ...), each holding that element's own fields -- a plain
`list` of scalars/arrays still becomes one dataset, as before. Order is
recovered on load by sorting those subgroup names back to int, not by
h5py's (insertion- or lexically-ordered) key iteration.

Escape hatches keep this from being a blind mirror of the pydantic
shape, on both the write and read side:

- `name_map` remaps a field name to a different (optionally nested,
  "/"-separated) path in the file, e.g. {"fit_result": "analysis/fit_result"}.
- `manual` hands a field's raw value to a caller-supplied writer (or,
  for loading, hands the caller-supplied reader the current group and
  takes back the field's value) instead of the generic dispatch, for
  fields whose on-disk shape doesn't fall out of their Python type
  (e.g. scan_ranges' dict[str, tuple[int, int]], flattened into paired
  *_start/*_end attributes) -- or, passed a no-op / omitted, to skip a
  field entirely (e.g. a duplicate reference field).
- `skip` (save) / `skip` + `extra` (load) drop a field from the generic
  walk. `skip` only applies to the call it's passed to, not to fields
  of the same name found while recursing into a nested model -- field
  names like "metadata" are reused with different meaning at different
  depths, so a skip can't safely propagate. `name_map`/`manual` do
  propagate: they're field-name -> behavior maps meant to apply
  wherever that field occurs. `extra` (load only) fills in a skipped,
  required field after the rest of the model has been resolved, e.g.
  from another field that was just loaded.
"""

from collections.abc import Callable
from datetime import datetime
from pydantic import BaseModel
from types import UnionType
from typing import Any, Union, get_args, get_origin
import h5py
import numpy as np

ManualWriter = Callable[[Any, h5py.Group], None]
ManualReader = Callable[[h5py.Group], Any]


def save_model(
    model: BaseModel,
    group: h5py.Group,
    name_map: dict[str, str] | None = None,
    manual: dict[str, ManualWriter] | None = None,
    skip: set[str] | None = None,
) -> None:
    """Write every non-None field of `model` into `group`.

    `skip` only applies to this call's own fields, not to fields of the
    same name found while recursing into a nested model -- field names
    like "metadata" are reused with different meaning at different
    depths, so a skip can't safely propagate. `name_map`/`manual` do
    propagate: they're field-name -> behavior maps that are meant to
    apply wherever that field occurs.
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


def load_model(
    model_cls: type[BaseModel],
    group: h5py.Group,
    name_map: dict[str, str] | None = None,
    manual: dict[str, ManualReader] | None = None,
    skip: set[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> BaseModel:
    """Build a `model_cls` instance by reading its fields out of `group`.

    See module docstring for `name_map`/`manual`/`skip`/`extra`. `extra`
    values that are callables are invoked with the dict of already-
    resolved fields (post generic walk, pre construction) and can
    reference another field's just-loaded value; non-callable values
    are used as-is.
    """
    name_map = name_map or {}
    manual = manual or {}
    skip = skip or set()
    extra = extra or {}

    resolved: dict[str, Any] = {}
    for field_name, field in model_cls.model_fields.items():
        if field_name in skip:
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


def _decode(value: Any) -> Any:
    return value.decode() if isinstance(value, bytes) else value


def _exists(group: h5py.Group, path: str) -> bool:
    return path in group.attrs or path in group


def _read(
    group: h5py.Group,
    path: str,
    annotation: Any,
    name_map: dict[str, str],
    manual: dict[str, ManualReader],
) -> Any:
    ann = _unwrap(annotation)
    origin = get_origin(ann)

    if isinstance(ann, type) and issubclass(ann, BaseModel):
        if path not in group:
            return None
        return load_model(ann, group[path], name_map, manual)

    if origin is dict:
        if path not in group:
            return None
        sub = group[path]
        args = get_args(ann)
        value_type = _unwrap(args[1]) if len(args) == 2 else Any
        is_model = isinstance(value_type, type) and issubclass(value_type, BaseModel)
        return {
            key: (
                load_model(value_type, sub[key], name_map, manual)
                if is_model
                else _read_leaf(sub, key)
            )
            for key in sub.keys()
        }

    if origin is list:
        args = get_args(ann)
        elem_type = _unwrap(args[0]) if args else Any
        elem_is_model = isinstance(elem_type, type) and issubclass(elem_type, BaseModel)
        elem_is_array = elem_type is np.ndarray
        if elem_is_model or elem_is_array:
            if path not in group:
                return None
            sub = group[path]
            if isinstance(sub, h5py.Dataset):
                # An empty list has nothing to inspect at write time to know
                # it was meant to hold BaseModel/ndarray elements, so it
                # falls back to the plain empty-dataset path there -- the
                # only thing that path can produce is an empty dataset, so
                # seeing one here unambiguously means "empty list".
                return []
            indexes = sorted(int(i) for i in sub.keys())
            if elem_is_model:
                return [
                    load_model(elem_type, sub[str(i)], name_map, manual)
                    for i in indexes
                ]
            return [sub[str(i)][()] for i in indexes]
        if path in group.attrs:
            return [_decode(v) for v in group.attrs[path]]
        if path in group:
            return [_decode(v) for v in group[path][()]]
        return None

    if origin is tuple:
        if not _exists(group, path):
            return None
        elem_type = get_args(ann)[0] if get_args(ann) else float
        arr = group.attrs[path] if path in group.attrs else group[path][()]
        return tuple(elem_type(v) for v in arr)

    if ann is datetime:
        if path not in group.attrs:
            return None
        return datetime.fromisoformat(group.attrs[path])

    if ann is np.ndarray:
        if path not in group:
            return None
        ds = group[path]
        arr = ds[()]
        if ds.attrs.get("none_as_nan"):
            return np.array([None if np.isnan(v) else v for v in arr], dtype=object)
        return arr

    if ann is Any:
        # No declared type to dispatch on (e.g. `metadata: SerializeAsAny[Any]`
        # holding an arbitrary dict, such as a nested model_dump()). Fall back
        # to inspecting the h5py node itself -- the same structure-driven walk
        # `_read_leaf` already does for dict[str, Any] values like raw_data,
        # just entered from a plain field instead of a dict item.
        if path in group.attrs:
            return _decode(group.attrs[path])
        if path in group:
            return _read_leaf(group, path)
        return None

    # plain scalar: str / int / float / bool
    if path not in group.attrs:
        return None
    value = group.attrs[path]
    if ann in (str, int, float, bool):
        return ann(value) if ann is not str else _decode(value)
    return value


def _read_leaf(group: h5py.Group, key: str) -> Any:
    """Structure-driven read for container-of-Any fields (e.g. raw_data,
    or a bare-`Any` field like `metadata`): a nested Group becomes a dict
    of leaves -- its attrs as scalar entries, its children (sub-groups /
    datasets) recursed the same way -- and a Dataset becomes an array, no
    type annotation available to guide the choice."""
    item = group[key]
    if isinstance(item, h5py.Group):
        result = {k: _decode(v) for k, v in item.attrs.items()}
        result.update({k: _read_leaf(item, k) for k in item.keys()})
        return result
    return item[()]


def _to_array(value: list | tuple) -> Any:
    """list/tuple -> something h5py's create_dataset accepts.

    Strings pass through as-is (h5py auto-picks a variable-length string
    dtype); everything else is handed to np.asarray unchanged.
    """
    seq = list(value)
    if seq and all(isinstance(v, str) for v in seq):
        return seq
    return np.asarray(seq)


def _unwrap(annotation: Any) -> Any:
    """Peel Optional[...] and Annotated[...] layers off a type annotation."""
    ann = annotation
    while True:
        if hasattr(ann, "__metadata__"):  # Annotated[X, ...]
            ann = get_args(ann)[0]
            continue
        origin = get_origin(ann)
        if origin is Union or origin is UnionType:
            args = [a for a in get_args(ann) if a is not type(None)]
            if len(args) == 1:
                ann = args[0]
                continue
        break
    return ann


def _write(
    group: h5py.Group,
    name: str,
    value: Any,
    name_map: dict[str, str],
    manual: dict[str, ManualWriter],
) -> None:
    if isinstance(value, BaseModel):
        save_model(value, group.create_group(name), name_map, manual)
    elif isinstance(value, dict):
        sub = group.create_group(name)
        for key, item in value.items():
            if item is None:
                continue
            _write(sub, key, item, name_map, manual)
    elif isinstance(value, datetime):
        group.attrs[name] = value.isoformat()
    elif isinstance(value, np.ndarray):
        if value.dtype == object:
            # h5py has no native representation for an object-dtype array
            # (e.g. one built from a tuple with a missing element, like
            # rms_sizes=(1.25, None) for a single-plane scan). None is the
            # only object value this ever legitimately holds, so it's
            # written as NaN, tagged so load_model can convert it back --
            # untagged NaN (from a plain float array) reads back as NaN.
            value = np.array([np.nan if v is None else v for v in value], dtype=float)
            group.create_dataset(name, data=value).attrs["none_as_nan"] = True
        else:
            group.create_dataset(name, data=value)
    elif isinstance(value, list) and value and isinstance(value[0], (BaseModel, np.ndarray)):
        # A list of BaseModel or ndarray elements gets one entry per index
        # instead of one flat dataset -- needed for BaseModel (can't live in
        # a dataset at all) and for ndarray specifically because a flat
        # dataset requires every element to share one shape (np.asarray of
        # a ragged list either raises or silently produces a useless
        # object-dtype array); index-keyed entries impose no such
        # constraint. Reuses _write itself per element, so it dispatches
        # BaseModel -> subgroup / ndarray -> dataset the normal way.
        sub = group.create_group(name)
        for i, elem in enumerate(value):
            _write(sub, str(i), elem, name_map, manual)
    elif isinstance(value, (list, tuple)):
        group.create_dataset(name, data=_to_array(value))
    elif isinstance(value, (str, int, float, bool, np.generic)):
        group.attrs[name] = value
    else:
        group.attrs[f"{name}_unsupported"] = str(value)
