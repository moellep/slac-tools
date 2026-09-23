import datetime
import h5py
import numpy
import pydantic
import pytest
import slac_tools.pydantic_h5
import typing


class Detector(pydantic.BaseModel):
    """A nested model -- exercises BaseModel fields, both singly
    (dict[str, Detector], list[Detector]) and as leaf scalars/arrays."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    label: str
    values: numpy.ndarray
    gain: float | None = None


class Sample(pydantic.BaseModel):
    """One model touching every field category pydantic_h5.py dispatches
    on: scalars, datetime, tuple, list[str], required/Optional ndarray
    (including an object-dtype one with a None entry), dict[str,
    BaseModel], list[BaseModel], list[ndarray] (including ragged), and a
    bare dict[str, Any] structure-driven blob."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    count: int
    ratio: float
    active: bool
    timestamp: datetime.datetime
    tags: list[str]
    coordinates: tuple[float, float]
    reading: numpy.ndarray
    rms_sizes: numpy.ndarray | None = None
    optional_reading: numpy.ndarray | None = None
    notes: str | None = None
    detectors: dict[str, Detector]
    scans: list[Detector]
    series: list[numpy.ndarray]
    raw_data: dict[str, typing.Any]


def test_empty_lists_and_back(tmp_path):
    s = _make_sample(scans=[], series=[])
    _assert_equal(s, _round_trip(s, tmp_path))


def test_fully_populated_optional_array_stays_float64(tmp_path):
    s = _make_sample(rms_sizes=numpy.array([1.0, 2.0, 3.0]))
    r = _round_trip(s, tmp_path)
    _assert_equal(s, r)
    assert r.rms_sizes.dtype == numpy.float64


def test_list_of_models_preserves_order_past_ten_elements(tmp_path):
    # h5py returns group keys in lexical order ("0","1","10","11","2",...),
    # not insertion order -- 11+ elements is what actually exercises that.
    s = _make_sample(
        scans=[
            Detector(label=f"scan{i}", values=numpy.array([float(i)]))
            for i in range(12)
        ]
    )
    _assert_equal(s, _round_trip(s, tmp_path))


def test_list_of_ragged_arrays(tmp_path):
    s = _make_sample(
        series=[
            numpy.array([1.0, 2.0, 3.0]),
            numpy.array([4.0]),
            numpy.array([5.0, 6.0, 7.0, 8.0]),
        ]
    )
    _assert_equal(s, _round_trip(s, tmp_path))


def test_name_map_manual_and_skip(tmp_path):
    # backward compatibility support: name_map, manual read/write and skip
    def _write_coordinates(value, group):
        group.attrs["coordinates"] = f"{value[0]},{value[1]}"

    def _read_coordinates(group):
        x, y = group.attrs["coordinates"].split(",")
        return (float(x), float(y))

    name_map = {"reading": "meta/reading"}
    s = _make_sample(optional_reading=numpy.array([9.9]))
    p = tmp_path / "out.h5"
    with h5py.File(p, "w") as f:
        slac_tools.pydantic_h5.save_model(
            s,
            f,
            name_map=name_map,
            manual={"coordinates": _write_coordinates},
            skip={"rms_sizes"},
        )
        assert "reading" not in f
        assert "meta/reading" in f
        assert "coordinates" not in f
        assert f.attrs["coordinates"] == "1.5,-2.5"
        assert "rms_sizes" not in f
        assert "optional_reading" in f
        assert f.attrs["count"] == 7

    with h5py.File(p, "r") as f:
        loaded = slac_tools.pydantic_h5.load_model(
            Sample,
            f,
            name_map=name_map,
            manual={"coordinates": _read_coordinates},
            skip={"optional_reading"},
            extra={
                "notes": lambda resolved: f"derived from {resolved['name']}",
                "count": 99,
            },
        )

    numpy.testing.assert_equal(loaded.reading, s.reading)
    assert loaded.coordinates == s.coordinates
    assert loaded.optional_reading is None
    assert loaded.notes == f"derived from {s.name}"
    assert loaded.count == 99


def test_nan_and_back(tmp_path):
    s = _make_sample(rms_sizes=numpy.array([1.25, numpy.nan]))
    _assert_equal(s, _round_trip(s, tmp_path))


def test_object_dtype_array_with_none_and_back(tmp_path):
    # dtype=object preserves None values
    s = _make_sample(rms_sizes=numpy.array([1.25, None], dtype=object))
    _assert_equal(s, _round_trip(s, tmp_path))


def test_optional_fields_omitted_when_none(tmp_path):
    s = _make_sample(optional_reading=None, notes=None, rms_sizes=None)
    _assert_equal(s, _round_trip(s, tmp_path))


def test_raw_data_includes_scalar_values(tmp_path):
    # dict[str, Any] values that land as a plain h5py attr (scalars) used
    # to be silently dropped on read -- only children (groups/datasets)
    # were enumerated, missing anything that was actually in group.attrs.
    s = _make_sample(
        raw_data={
            "count": 3,
            "label": "wire-scan",
            "nested": {"x": numpy.array([1.0])},
        }
    )
    _assert_equal(s, _round_trip(s, tmp_path))


def test_round_trip_all_fields(tmp_path):
    s = _make_sample()
    _assert_equal(s, _round_trip(s, tmp_path))


def test_unsupported_type_raises(tmp_path):
    # raw_data: dict[str, Any] takes any value, bypassing pydantic
    # validation, so a set (no generic dispatch in _write) reaches the
    # module's catch-all
    s = _make_sample(raw_data={"bad": {1, 2, 3}})
    with pytest.raises(TypeError, match="bad"):
        _round_trip(s, tmp_path)


def _assert_equal(original: pydantic.BaseModel, loaded: pydantic.BaseModel) -> None:
    assert original is not loaded
    # model_dump() is generic, order/path-aware of whole tree
    numpy.testing.assert_equal(loaded.model_dump(), original.model_dump())


def _make_sample(**overrides) -> Sample:
    defaults = dict(
        name="scan-001",
        count=7,
        ratio=0.125,
        active=True,
        timestamp=datetime.datetime(
            2026, 5, 21, 3, 3, 57, tzinfo=datetime.timezone.utc
        ),
        tags=["x", "y", "u"],
        coordinates=(1.5, -2.5),
        reading=numpy.array([1.0, 2.0, 3.0]),
        rms_sizes=numpy.array([353.5, 212.1]),
        optional_reading=None,
        notes="hand-written test fixture",
        detectors={
            "PMT29150": Detector(
                label="PMT29150", values=numpy.array([1.0, 2.0]), gain=1.5
            ),
            "PMT756": Detector(label="PMT756", values=numpy.array([3.0, 4.0, 5.0])),
        },
        scans=[
            Detector(label=f"scan{i}", values=numpy.array([float(i)]))
            for i in range(11)
        ],
        series=[numpy.array([1.0, 2.0, 3.0]), numpy.array([4.0, 5.0])],
        raw_data={
            "WIRE:TEST:100": numpy.array([10.0, 20.0]),
            "BPM27201": {"x": numpy.array([0.1, 0.2]), "y": numpy.array([0.3, 0.4])},
        },
    )
    defaults.update(overrides)
    return Sample(**defaults)


def _round_trip(model: pydantic.BaseModel, tmp_path) -> pydantic.BaseModel:
    path = tmp_path / "out.h5"
    with h5py.File(path, "w") as f:
        slac_tools.pydantic_h5.save_model(model, f)
    with h5py.File(path, "r") as f:
        return slac_tools.pydantic_h5.load_model(type(model), f)
