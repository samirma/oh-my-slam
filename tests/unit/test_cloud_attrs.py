"""The point-cloud attributes of §2.2: defaults, parsing, validation and scopes (one definition)."""

from __future__ import annotations

import math

import pytest

from oh_my_slam.core.cloud_attrs import (
    ATTRIBUTES,
    KEYS,
    CloudAttrs,
    CloudScope,
    help_text,
    parse_cloud_attrs,
)
from oh_my_slam.core.errors import ExitCode, UsageError

IMAGE, MAP, SEGMENT = CloudScope.IMAGE, CloudScope.MAP, CloudScope.SEGMENT
PIXEL_KEYS = ("stride", "min-depth", "max-depth", "edge")


def test_attribute_table_matches_the_spec() -> None:
    assert KEYS == ("color", "stride", "min-depth", "max-depth", "edge", "voxel", "normals",
                    "label", "encoding")
    assert tuple(a.key for a in ATTRIBUTES if a.pixel_level) == PIXEL_KEYS


def test_defaults_per_scope() -> None:
    d = CloudAttrs.defaults(IMAGE)
    assert d == CloudAttrs() == parse_cloud_attrs(None, IMAGE) == parse_cloud_attrs("", IMAGE)
    assert (d.color, d.stride, d.min_depth, d.max_depth, d.edge, d.voxel, d.normals, d.label,
            d.encoding) == ("rgb", 1, 0.0, math.inf, 0.04, 0.0, False, False, "binary")
    assert CloudAttrs.defaults(MAP) == d
    assert parse_cloud_attrs(None, IMAGE | SEGMENT) == CloudAttrs(color="segment")
    assert parse_cloud_attrs(None, MAP | SEGMENT) == CloudAttrs(color="segment")


def test_parse_spec_example_and_every_key() -> None:
    a = parse_cloud_attrs("color=segment,voxel=0.01,normals=on", IMAGE)
    assert a == CloudAttrs(color="segment", voxel=0.01, normals=True)
    b = parse_cloud_attrs("color=height, stride=3,min-depth=0.5,max-depth=4,edge=0,voxel=0.02,"
                          "normals=on,label=on,encoding=ascii", IMAGE)
    assert b == CloudAttrs("height", 3, 0.5, 4.0, 0.0, 0.02, True, True, "ascii")
    # several -p options and a mapping (the viewer's controls) are merged the same way
    assert parse_cloud_attrs(["color=none", "label=on"], IMAGE) == CloudAttrs(color="none",
                                                                              label=True)
    assert parse_cloud_attrs({"voxel": "0.1", "max-depth": "inf"}, IMAGE) == CloudAttrs(voxel=0.1)
    assert parse_cloud_attrs({"voxel": "0.1"}, MAP) == CloudAttrs(voxel=0.1)


@pytest.mark.parametrize(("spec", "message"), [
    ("colour=rgb", "unknown point-cloud attribute 'colour'"),
    ("color=red", "color must be rgb|segment|height|none"),
    ("stride=0", "stride must be an integer >= 1"),
    ("stride=1.5", "stride must be"),
    ("stride=abc", "stride must be"),
    ("min-depth=-1", "min-depth must be metres >= 0"),
    ("min-depth=inf", "min-depth must be"),
    ("max-depth=0", "max-depth must be metres > 0"),
    ("max-depth=-inf", "max-depth must be"),
    ("edge=-0.1", "edge must be a relative depth jump >= 0"),
    ("voxel=-1", "voxel must be metres >= 0"),
    ("voxel=nan", "voxel must be"),
    ("voxel=", "voxel must be"),
    ("normals=yes", "normals must be on|off"),
    ("label=1", "label must be on|off"),
    ("encoding=utf8", "encoding must be binary|ascii"),
    ("min-depth=3,max-depth=2", "min-depth (3) must be smaller than max-depth (2)"),
    ("voxel=0.1,voxel=0.2", "given twice"),
    ("voxel", "is not key=value"),
])
def test_bad_values_are_actionable_usage_errors(spec: str, message: str) -> None:
    with pytest.raises(UsageError) as e:
        parse_cloud_attrs(spec, IMAGE)
    assert message in str(e.value)
    assert e.value.exit_code == ExitCode.USAGE


@pytest.mark.parametrize("key", PIXEL_KEYS)
def test_map_scope_refuses_pixel_level_attributes(key: str) -> None:
    value = {"stride": "2", "min-depth": "1", "max-depth": "3", "edge": "0"}[key]
    parse_cloud_attrs(f"{key}={value}", IMAGE)  # fine for a single image
    for scope in (MAP, MAP | SEGMENT):
        with pytest.raises(UsageError, match="pixel-level attribute"):
            parse_cloud_attrs(f"{key}={value}", scope)
    # the unknown-key hint for a map lists only the keys that apply
    with pytest.raises(UsageError) as e:
        parse_cloud_attrs("nope=1", MAP)
    assert "stride" not in str(e.value) and "voxel" in str(e.value)


def test_segment_scope_fixes_color() -> None:
    assert parse_cloud_attrs("color=segment,voxel=0.05", IMAGE | SEGMENT).voxel == 0.05
    for other in ("rgb", "height", "none"):
        with pytest.raises(UsageError, match="color is fixed to segment"):
            parse_cloud_attrs(f"color={other}", IMAGE | SEGMENT)


def test_describe_records_every_applicable_attribute_and_reads_back() -> None:
    assert CloudAttrs().describe(IMAGE) == ("color=rgb,stride=1,min-depth=0,max-depth=inf,"
                                            "edge=0.04,voxel=0,normals=off,label=off,"
                                            "encoding=binary")
    assert CloudAttrs(color="segment").describe(MAP) == ("color=segment,voxel=0,normals=off,"
                                                         "label=off,encoding=binary")
    a = CloudAttrs("height", 3, 0.25, 7.5, 0.0, 0.015, True, True, "ascii")
    assert parse_cloud_attrs(a.describe(IMAGE), IMAGE) == a
    m = CloudAttrs(color="none", voxel=0.2, normals=True)
    assert parse_cloud_attrs(m.describe(MAP), MAP) == m


def test_help_text_lists_keys_and_defaults() -> None:
    h = help_text(IMAGE)
    assert all(k in h for k in KEYS) and "edge=" in h and "[0.04]" in h and "[rgb]" in h
    hm = help_text(MAP | SEGMENT)
    assert "stride" not in hm and "segment (fixed)" in hm and "[segment]" in hm


def test_scope_needs_exactly_one_source_kind() -> None:
    with pytest.raises(ValueError):
        parse_cloud_attrs(None, SEGMENT)
    with pytest.raises(ValueError):
        parse_cloud_attrs(None, IMAGE | MAP)
