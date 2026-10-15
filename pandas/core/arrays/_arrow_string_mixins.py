from __future__ import annotations

from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
)
import unicodedata

import numpy as np

from pandas._libs import (
    arrow_regex,
    lib,
)
from pandas.compat import (
    HAS_PYARROW,
    pa_version_under17p0,
    pa_version_under21p0,
)

from pandas.core.strings.object_array import compile_match_pattern

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.compute as pc

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        Scalar,
        npt,
    )


def _py_buffer(arr: np.ndarray | None) -> pa.Buffer | None:
    return None if arr is None else pa.py_buffer(arr)


def _string_array(
    pa_type: pa.DataType,
    length: int,
    buffers: tuple[npt.NDArray[np.uint8] | None, np.ndarray, npt.NDArray[np.uint8]],
) -> pa.Array:
    validity, offsets, data = buffers
    return pa.Array.from_buffers(
        pa_type,
        length,
        [_py_buffer(validity), pa.py_buffer(offsets), pa.py_buffer(data)],
    )


class ArrowStringArrayMixin:
    _pa_array: pa.ChunkedArray

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _from_pyarrow_array(self, pa_array: pa.Array | pa.ChunkedArray) -> Self:
        raise NotImplementedError

    def _convert_bool_result(self, result, na=lib.no_default, method_name=None):
        # Convert a bool-dtype result to the appropriate result type
        raise NotImplementedError

    def _convert_int_result(self, result):
        # Convert an integer-dtype result to the appropriate result type
        raise NotImplementedError

    def _apply_elementwise(self, func: Callable[..., Any]) -> list[list[Any]]:
        raise NotImplementedError

    def _re_bool(self, func: Callable[[str], Any]) -> pa.ChunkedArray:
        # whether `func`, a method of a compiled pattern, finds a match
        chunks = []
        for chunk in self._pa_array.iterchunks():
            validity, values = arrow_regex.search(chunk, func)
            chunks.append(
                pa.Array.from_buffers(
                    pa.bool_(),
                    len(chunk),
                    [_py_buffer(validity), pa.py_buffer(values)],
                )
            )
        return pa.chunked_array(chunks, type=pa.bool_())

    def _re_count(self, func: Callable[[str], list]) -> pa.ChunkedArray:
        # the width pyarrow's count_substring_regex reports for this storage type
        pa_type = (
            pa.int64() if pa.types.is_large_string(self._pa_array.type) else pa.int32()
        )
        chunks = []
        for chunk in self._pa_array.iterchunks():
            validity, values = arrow_regex.count(chunk, func)
            values = values.astype(pa_type.to_pandas_dtype(), copy=False)
            chunks.append(
                pa.Array.from_buffers(
                    pa_type,
                    len(chunk),
                    [_py_buffer(validity), pa.py_buffer(values)],
                )
            )
        return pa.chunked_array(chunks, type=pa_type)

    def _re_map_str(self, func: Callable[[str], str | None]) -> pa.ChunkedArray:
        pa_type = self._pa_array.type
        large = pa.types.is_large_string(pa_type)
        chunks = [
            _string_array(pa_type, len(chunk), arrow_regex.map_str(chunk, func, large))
            for chunk in self._pa_array.iterchunks()
        ]
        return pa.chunked_array(chunks, type=pa_type)

    def _re_extract(self, compiled: re.Pattern[str]) -> list[pa.ChunkedArray]:
        # one array per group of `compiled`
        pa_type = self._pa_array.type
        large = pa.types.is_large_string(pa_type)
        per_chunk = [
            [
                _string_array(pa_type, len(chunk), buffers)
                for buffers in arrow_regex.extract(
                    chunk, compiled.search, compiled.groups, large
                )
            ]
            for chunk in self._pa_array.iterchunks()
        ]
        return [
            pa.chunked_array([chunks[k] for chunks in per_chunk], type=pa_type)
            for k in range(compiled.groups)
        ]

    def _re_map_str_list(
        self, func: Callable[[str], list[str | None]]
    ) -> pa.ChunkedArray:
        pa_type = self._pa_array.type
        large = pa.types.is_large_string(pa_type)
        chunks = []
        for chunk in self._pa_array.iterchunks():
            validity, list_offsets, child_buffers = arrow_regex.map_str_list(
                chunk, func, large
            )
            child = _string_array(pa_type, len(child_buffers[1]) - 1, child_buffers)
            chunks.append(
                pa.Array.from_buffers(
                    pa.list_(pa_type),
                    len(chunk),
                    [_py_buffer(validity), pa.py_buffer(list_offsets)],
                    children=[child],
                )
            )
        return pa.chunked_array(chunks, type=pa.list_(pa_type))

    def _str_len(self):
        result = pc.utf8_length(self._pa_array)
        return self._convert_int_result(result)

    def _str_lower(self) -> Self:
        return self._from_pyarrow_array(pc.utf8_lower(self._pa_array))

    def _str_upper(self) -> Self:
        return self._from_pyarrow_array(pc.utf8_upper(self._pa_array))

    def _str_strip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._pa_array)
        else:
            result = pc.utf8_trim(self._pa_array, characters=to_strip)
        return self._from_pyarrow_array(result)

    def _str_lstrip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_ltrim(self._pa_array, characters=to_strip)
        return self._from_pyarrow_array(result)

    def _str_rstrip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_rtrim(self._pa_array, characters=to_strip)
        return self._from_pyarrow_array(result)

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ) -> Self:
        if width < 0:
            width = 0
        if side == "left":
            pa_pad = pc.utf8_lpad
        elif side == "right":
            pa_pad = pc.utf8_rpad
        elif side == "both":
            if pa_version_under17p0:
                # GH#59624 fall back to object dtype
                from pandas import array

                obj_arr = self.astype(object, copy=False)  # type: ignore[attr-defined]
                obj = array(obj_arr, dtype=object)
                result = obj._str_pad(width, side, fillchar)  # type: ignore[attr-defined]
                return type(self)._from_sequence(result, dtype=self.dtype)  # type: ignore[attr-defined]
            else:
                # GH#54792
                # https://github.com/apache/arrow/issues/15053#issuecomment-2317032347
                lean_left = (width % 2) == 0
                pa_pad = partial(pc.utf8_center, lean_left_on_odd_padding=lean_left)
        else:
            raise ValueError(
                f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'"
            )
        return self._from_pyarrow_array(
            pa_pad(self._pa_array, width=width, padding=fillchar)
        )

    def _str_zfill(self, width: int) -> Self:
        if pa_version_under21p0:
            predicate = lambda val: val.zfill(width)
            result = self._apply_elementwise(predicate)
            return self._from_pyarrow_array(
                pa.chunked_array(result, type=self._pa_array.type)
            )
        # pc.utf8_zfill raises on a negative width, while str.zfill returns the
        # string unchanged -> clamping to zero to get that behaviour (GH#69486)
        width = max(width, 0)
        return self._from_pyarrow_array(pc.utf8_zfill(self._pa_array, width))

    def _str_normalize(self, form: Literal["NFC", "NFD", "NFKC", "NFKD"]) -> Self:
        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError("invalid normalization form")
        if form in ("NFC", "NFKC"):
            # GH#64359 pc.utf8_normalize only decomposes; it skips the canonical
            #  composition step, so for the composing forms it returns decomposed
            #  output. Fall back to unicodedata for these.
            predicate = lambda val: unicodedata.normalize(form, val)
            result = self._apply_elementwise(predicate)
            return self._from_pyarrow_array(
                pa.chunked_array(result, type=self._pa_array.type)
            )
        return self._from_pyarrow_array(pc.utf8_normalize(self._pa_array, form=form))

    def _str_get(self, i: int) -> Self:
        lengths = pc.utf8_length(self._pa_array)
        if i >= 0:
            out_of_bounds = pc.greater_equal(i, lengths)
            start = i
            stop = i + 1
            step = 1
        else:
            out_of_bounds = pc.greater(-i, lengths)
            start = i
            stop = i - 1
            step = -1
        not_out_of_bounds = pc.invert(out_of_bounds.fill_null(True))
        selected = pc.utf8_slice_codeunits(
            self._pa_array, start=start, stop=stop, step=step
        )
        null_value = pa.scalar(None, type=self._pa_array.type)
        result = pc.if_else(not_out_of_bounds, selected, null_value)
        return self._from_pyarrow_array(result)

    def _str_slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ) -> Self:
        if start is None:
            if step is not None and step < 0:
                # GH#59710
                start = -1
            else:
                start = 0
        if step is None:
            step = 1
        return self._from_pyarrow_array(
            pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step)
        )

    def _str_getitem(self, key: slice | int) -> Self:
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    def _str_slice_replace(
        self, start: int | None = None, stop: int | None = None, repl: str | None = None
    ) -> Self:
        if repl is None:
            repl = ""
        if start is None:
            start = 0
        if stop is None:
            stop = np.iinfo(np.int64).max
        return self._from_pyarrow_array(
            pc.utf8_replace_slice(self._pa_array, start, stop, repl)
        )

    def _str_replace(
        self,
        pat: str | re.Pattern[str],
        repl: str | Callable[..., Any],
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Self:
        if not case:
            flags |= re.IGNORECASE

        if regex or flags or callable(repl):
            # as in ObjectStringArrayMixin._str_replace, an already-compiled
            #  `pat` keeps its own flags
            if not isinstance(pat, re.Pattern):
                if not regex:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)
            if n > 0:
                func = partial(pat.sub, repl, count=n)
            else:
                # a keyword argument makes each call noticeably slower
                func = partial(pat.sub, repl)
            return self._from_pyarrow_array(self._re_map_str(func))

        if pat == "":
            # pyarrow hangs for empty patterns
            # (https://github.com/apache/arrow/issues/39149)
            # use same func definition as ObjectStringArrayMixin._str_replace
            func = lambda val: val.replace(pat, repl, n)
            return self._from_pyarrow_array(self._re_map_str(func))

        # https://github.com/apache/arrow/issues/39149
        # GH 56404, unexpected behavior with negative max_replacements with pyarrow.
        pa_max_replacements = None if n < 0 else n
        result = pc.replace_substring(
            self._pa_array,
            pattern=pat,
            replacement=repl,
            max_replacements=pa_max_replacements,
        )
        return self._from_pyarrow_array(result)

    def _str_capitalize(self) -> Self:
        return self._from_pyarrow_array(pc.utf8_capitalize(self._pa_array))

    def _str_title(self) -> Self:
        return self._from_pyarrow_array(pc.utf8_title(self._pa_array))

    def _str_swapcase(self) -> Self:
        return self._from_pyarrow_array(pc.utf8_swapcase(self._pa_array))

    def _str_removeprefix(self, prefix: str):
        if prefix == "":
            return self._from_pyarrow_array(self._pa_array)
        starts_with = pc.starts_with(self._pa_array, pattern=prefix)
        removed = pc.utf8_slice_codeunits(self._pa_array, len(prefix))
        result = pc.if_else(starts_with, removed, self._pa_array)
        return self._from_pyarrow_array(result)

    def _str_removesuffix(self, suffix: str):
        if suffix == "":
            return self._from_pyarrow_array(self._pa_array)
        ends_with = pc.ends_with(self._pa_array, pattern=suffix)
        removed = pc.utf8_slice_codeunits(self._pa_array, 0, stop=-len(suffix))
        result = pc.if_else(ends_with, removed, self._pa_array)
        return self._from_pyarrow_array(result)

    def _str_startswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ):
        if isinstance(pat, str):
            result = pc.starts_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            # For empty tuple we return null for missing values and False
            #  for valid values.
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.starts_with(self._pa_array, pattern=pat[0])

            for p in pat[1:]:
                result = pc.or_(result, pc.starts_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name="startswith")

    def _str_endswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ):
        if isinstance(pat, str):
            result = pc.ends_with(self._pa_array, pattern=pat)
        elif len(pat) == 0:
            # For empty tuple we return null for missing values and False
            #  for valid values.
            result = pc.if_else(pc.is_null(self._pa_array), None, False)
        else:
            result = pc.ends_with(self._pa_array, pattern=pat[0])

            for p in pat[1:]:
                result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name="endswith")

    def _str_isalnum(self):
        result = pc.utf8_is_alnum(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isalpha(self):
        result = pc.utf8_is_alpha(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isascii(self):
        result = pc.string_is_ascii(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdecimal(self):
        result = pc.utf8_is_decimal(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdigit(self):
        if pa_version_under21p0:
            # https://github.com/pandas-dev/pandas/issues/61466
            res_list = self._apply_elementwise(str.isdigit)
            return self._convert_bool_result(
                pa.chunked_array(res_list, type=pa.bool_())
            )
        result = pc.utf8_is_digit(self._pa_array)
        return self._convert_bool_result(result)

    def _str_islower(self):
        result = pc.utf8_is_lower(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isnumeric(self):
        result = pc.utf8_is_numeric(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isspace(self):
        result = pc.utf8_is_space(self._pa_array)
        return self._convert_bool_result(result)

    def _str_istitle(self):
        result = pc.utf8_is_title(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isupper(self):
        result = pc.utf8_is_upper(self._pa_array)
        return self._convert_bool_result(result)

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
        regex: bool = True,
    ):
        if regex:
            if not case:
                flags |= re.IGNORECASE
            result = self._re_bool(re.compile(pat, flags=flags).search)
        else:
            result = pc.match_substring(self._pa_array, pat, ignore_case=not case)
        return self._convert_bool_result(result, na=na, method_name="contains")

    def _str_match(
        self,
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        result = self._re_bool(compile_match_pattern(pat, case, flags).match)
        return self._convert_bool_result(result, na=na, method_name="match")

    def _str_fullmatch(
        self,
        pat: str | re.Pattern[str],
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if not case:
            flags |= re.IGNORECASE
        result = self._re_bool(re.compile(pat, flags=flags).fullmatch)
        return self._convert_bool_result(result, na=na, method_name="fullmatch")

    def _str_count(self, pat: str | re.Pattern[str], flags: int = 0):
        result = self._re_count(re.compile(pat, flags=flags).findall)
        return self._convert_int_result(result)

    def _str_find(self, sub: str, start: int = 0, end: int | None = None):
        # min_count=0 so that an empty or all-null array reports True instead of
        #  null, keeping it on the pyarrow path below
        if not pc.all(pc.string_is_ascii(self._pa_array), min_count=0).as_py():
            # GH#64123 - pc.find_substring returns byte offsets instead of
            # character offsets for multi-byte UTF-8 characters, so we fall back
            # to Python str.find which correctly returns character offsets.
            res_list = self._apply_elementwise(lambda val: val.find(sub, start, end))
            return self._convert_int_result(pa.chunked_array(res_list, type=pa.int64()))

        if (start == 0 or start is None) and end is None:
            result = pc.find_substring(self._pa_array, sub)
        else:
            if sub == "":
                # GH#56792
                res_list = self._apply_elementwise(
                    lambda val: val.find(sub, start, end)
                )
                return self._convert_int_result(
                    pa.chunked_array(res_list, type=pa.int64())
                )
            if start is None:
                start_offset = 0
                start = 0
            elif start < 0:
                start_offset = pc.add(start, pc.utf8_length(self._pa_array))
                start_offset = pc.if_else(pc.less(start_offset, 0), 0, start_offset)
            else:
                start_offset = start
            slices = pc.utf8_slice_codeunits(self._pa_array, start, stop=end)
            result = pc.find_substring(slices, sub)
            found = pc.not_equal(result, pa.scalar(-1, type=result.type))
            offset_result = pc.add(result, start_offset)
            result = pc.if_else(found, offset_result, -1)
        result = result.cast(pa.int64())
        return self._convert_int_result(result)

    def _str_partition_expand(self, sep: str) -> pa.ChunkedArray:
        """
        Split each string on the first occurrence of ``sep``.

        Returns a ``list<string>`` array holding one three-element list per row
        -- the part before the separator, the separator, and the part after --
        which ``StringMethods._wrap_result`` expands into three columns. Rows
        without ``sep`` get two empty strings, matching ``str.partition``.

        The caller wraps this in an :class:`ArrowExtensionArray`; the rows are
        lists, so it is not of the calling array's own type.
        """
        if not sep:
            # pyarrow reports this as "Empty separator"; keep str.partition's
            #  wording so every dtype raises the same way
            raise ValueError("empty separator")

        str_type = self._pa_array.type
        chunks = [
            self._partition_chunk(chunk, sep, str_type)
            for chunk in self._pa_array.chunks
        ]
        return pa.chunked_array(chunks, type=pa.list_(str_type))

    @staticmethod
    def _partition_chunk(chunk: pa.Array, sep: str, str_type: pa.DataType) -> pa.Array:
        """
        Build the ``list<string>`` rows for one chunk of :meth:`_str_partition_expand`.

        Working a chunk at a time keeps the concatenation below within the
        offset width of ``str_type``, which matters for columns near the 2 GiB
        limit of 32-bit ``string``.
        """
        # max_splits=1 gives [before] when sep is absent, else [before, after];
        #  padding to a fixed two elements makes the tail null in the first case
        split = pc.split_pattern(chunk, sep, max_splits=1)
        pieces = pc.list_slice(split, 0, 2, return_fixed_size_list=True)

        before = pc.list_element(pieces, 0)
        tail = pc.list_element(pieces, 1)
        after = pc.fill_null(tail, pa.scalar("", type=str_type))
        middle = pc.if_else(
            pc.is_valid(tail),
            pa.scalar(sep, type=str_type),
            pa.scalar("", type=str_type),
        )

        n = len(chunk)
        # Interleave the three columns into [before[0], sep[0], after[0],
        #  before[1], ...]. Arrow has no kernel that zips arrays into rows, so
        #  take from their concatenation. Both ranges below are plain index
        #  arithmetic: NumPy builds them in one strided pass, while pyarrow has
        #  no arange at all before 21.0 and would need four kernels for the
        #  interleave, so they stay NumPy at every supported version.
        values = pa.concat_arrays([before, middle, after])
        indices = np.arange(3 * n, dtype=np.int64).reshape(3, n).T.reshape(-1)
        values = values.take(pa.array(indices))
        # int32 rather than int64 offsets, but built wide so that a chunk with
        #  more than 2**31 / 3 rows raises instead of wrapping around
        offsets = pa.array(np.arange(0, 3 * n + 1, 3, dtype=np.int64), type=pa.int32())
        # a null string partitions to a null row, not to a row of nulls
        mask = pc.is_null(before) if before.null_count else None
        return pa.ListArray.from_arrays(offsets, values, mask=mask)
