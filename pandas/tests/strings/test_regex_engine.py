"""
The regex methods of pyarrow-backed strings evaluate patterns with ``re``, so they
must agree with object dtype, including on patterns and data for which pyarrow's
RE2 kernels would give a different result.
"""

import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm

pa = pytest.importorskip("pyarrow")


@pytest.fixture(
    params=[
        pd.StringDtype("pyarrow", na_value=np.nan),
        pd.StringDtype("pyarrow"),
        pd.ArrowDtype(pa.string()),
        pd.ArrowDtype(pa.large_string()),
    ],
    ids=["str[pyarrow]", "string[pyarrow]", "string_arrow", "large_string_arrow"],
)
def arrow_string_dtype(request):
    return request.param


def _cast_like(expected, result):
    # `expected` is the object dtype result
    if isinstance(expected, pd.DataFrame):
        return expected.astype(result.dtypes.to_dict())
    if isinstance(result.dtype, pd.ArrowDtype) and pa.types.is_list(
        result.dtype.pyarrow_dtype
    ):
        values = pa.array(expected.tolist(), type=result.dtype.pyarrow_dtype)
        return pd.Series(values, index=expected.index, dtype=result.dtype)
    return expected.astype(result.dtype)


DATA = [
    "abc",
    "",
    "héllo wörld",
    "bar\n",
    "a\nb\n",
    "straße",
    "x١٢٣y",  # noqa: RUF001 ARABIC-INDIC digits
    "a b",  # # noqa: RUF001 NO-BREAK SPACE
    "a\x1cb",  # FILE SEPARATOR
    "ǅx",
    "K",  # KELVIN SIGN
]


@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        # \w, \d, \s and \b are Unicode-aware in re, ASCII-only in RE2
        ("contains", (r"\w+$",), {}),
        ("contains", (r"\d",), {}),
        ("contains", (r"a\sb",), {}),
        ("contains", (r"o\b",), {}),
        ("contains", (r"\w+",), {"flags": re.ASCII}),
        # $ matches before a trailing newline in re
        ("contains", (r"bar$",), {}),
        ("contains", (r"b$",), {"flags": re.MULTILINE}),
        # syntax RE2 does not support
        ("contains", (r"(?<=h)é",), {}),
        ("replace", (r"(\w)\1", "#"), {"regex": True}),
        ("contains", (r"(?=.*o).*w",), {}),
        # case folding
        ("contains", ("k",), {"case": False}),
        ("contains", ("ǆ",), {"case": False}),
        ("match", (r"\w\w",), {}),
        ("match", (r"\d",), {"flags": re.IGNORECASE}),
        ("match", (re.compile(r"^b", re.MULTILINE),), {}),
        ("fullmatch", (r"\w+",), {}),
        ("fullmatch", (r"bar$",), {}),
        ("fullmatch", (r".*\Z",), {}),
        ("fullmatch", (r"a.b.",), {"flags": re.DOTALL}),
        # empty matches are stepped over by character in re
        ("count", (r"",), {}),
        ("count", (r"\b",), {}),
        ("count", (r"\w",), {}),
        ("count", (r"(?<=l)",), {}),
        ("replace", (r"", "-"), {"regex": True}),
        ("replace", (r"$", "!"), {"regex": True}),
        ("replace", (r"\b", "|"), {"regex": True}),
        ("replace", (r"(\w)(\w)", r"\2\1"), {"regex": True}),
        ("replace", (r"(?P<first>\w)", r"<\g<first>>"), {"regex": True}),
        ("replace", (r"\w", "_"), {"n": 2, "regex": True}),
        ("replace", (r"\w", "_"), {"n": 0, "regex": True}),
        ("replace", ("L", "!"), {"case": False, "regex": True}),
        ("replace", ("L", "!"), {"case": False, "regex": False}),
        ("replace", (r"\w+", lambda m: m.group(0)[::-1]), {"regex": True}),
        ("replace", (re.compile(r"^\w", re.MULTILINE), "#"), {"regex": True}),
        ("split", (r"\s",), {"regex": True}),
        ("split", (r"(l)|\d",), {"regex": True}),
        ("split", (r"\w",), {"regex": True, "n": 1}),
        ("findall", (r"\d",), {}),
        ("findall", (r"\w+",), {}),
        ("findall", (r"(o)",), {}),
        ("extract", (r"(\w)(\d)?",), {"expand": True}),
        ("extract", (r"(?P<letter>\w)(?P<digit>\d)?",), {"expand": True}),
        ("extract", (r"(?P<letter>\w)",), {"expand": False}),
        ("extract", (r"(\s)",), {"expand": False}),
    ],
)
def test_regex_methods_match_object(arrow_string_dtype, method, args, kwargs):
    expected = getattr(pd.Series(DATA, dtype=object).str, method)(*args, **kwargs)
    result = getattr(pd.Series(DATA, dtype=arrow_string_dtype).str, method)(
        *args, **kwargs
    )
    expected = _cast_like(expected, result)
    if isinstance(expected, pd.DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat", [r"\p{L}", r"[\x{0e00}-\x{0e7f}]", r"(?P=undefined)", r"a{,2"]
)
def test_invalid_python_pattern_raises(arrow_string_dtype, pat):
    # RE2-only syntax is rejected as for object dtype, rather than evaluated
    ser = pd.Series(["a"], dtype=arrow_string_dtype)
    try:
        re.compile(pat)
    except re.error:
        with pytest.raises(re.error):
            ser.str.contains(pat)
    else:
        result = ser.str.contains(pat)
        expected = pd.Series(["a"], dtype=object).str.contains(pat)
        tm.assert_series_equal(result, expected.astype(result.dtype))


@pytest.mark.parametrize("pa_type", [pa.string(), pa.large_string()])
@pytest.mark.parametrize(
    "method, args, kwargs",
    [
        ("contains", (r"\w$",), {}),
        ("count", (r"\w",), {}),
        ("replace", (r"(\w)", r"\1\1"), {"regex": True}),
        ("replace", (r"x", "y"), {"regex": True}),
        ("split", (r"\s",), {"regex": True}),
        ("findall", (r"\w",), {}),
        ("extract", (r"(\w)(\d)?",), {"expand": True}),
    ],
)
def test_regex_methods_chunk_layouts(pa_type, method, args, kwargs):
    # the values are read from the arrow buffers directly, so cover offset
    #  (sliced) chunks, several chunks, nulls and empty chunks
    values = ["ab", None, "é1 ü", "", None, "z", "日本 語2"] * 3
    base = pa.array(values, type=pa_type)
    chunked = pa.chunked_array(
        [base.slice(1, 6), base.slice(0, 0), base.slice(9), pa.nulls(2, pa_type)],
        type=pa_type,
    )
    ser = pd.Series(pd.arrays.ArrowExtensionArray(chunked))
    expected = getattr(pd.Series(chunked.to_pylist(), dtype=object).str, method)(
        *args, **kwargs
    )
    result = getattr(ser.str, method)(*args, **kwargs)
    if isinstance(expected, pd.DataFrame):
        expected = expected.astype(result.dtypes.to_dict())
        tm.assert_frame_equal(result, expected)
    else:
        if isinstance(result.dtype, pd.ArrowDtype) and pa.types.is_list(
            result.dtype.pyarrow_dtype
        ):
            expected = pd.Series(
                pa.array(
                    [None if isinstance(x, float) else x for x in expected],
                    type=result.dtype.pyarrow_dtype,
                ),
                dtype=result.dtype,
            )
        else:
            expected = expected.astype(result.dtype)
        tm.assert_series_equal(result, expected)


def test_replace_callable_returning_non_str():
    ser = pd.Series(["a"], dtype=pd.ArrowDtype(pa.string()))
    with pytest.raises(TypeError, match="expected str instance, int found"):
        ser.str.replace("a", lambda m: 1, regex=True)
