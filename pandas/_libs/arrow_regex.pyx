"""
Evaluate Python ``re`` patterns on the values of pyarrow string arrays.

The kernels read the validity bitmap, offsets and UTF-8 data of a ``string`` or
``large_string`` array directly from its buffers, decode each valid value to a
``str``, call a method of a compiled ``re.Pattern`` on it, and write the result
into the buffers of a new Arrow array. No intermediate list or object ndarray of
the values is created, and the results are not converted from Python objects by
pyarrow.

All kernels take a single ``pyarrow.Array`` (one chunk) and return numpy arrays
that the caller wraps with ``pyarrow.Array.from_buffers``. Bitmaps are returned
packed, in Arrow's bit order. A validity bitmap is None when there are no nulls.
"""
from cpython.unicode cimport PyUnicode_DecodeUTF8
from cython cimport Py_ssize_t
from libc.stdint cimport (
    INT32_MAX,
    int32_t,
    int64_t,
    uint8_t,
    uintptr_t,
)
from libc.string cimport memcpy

import numpy as np

cimport numpy as cnp

cnp.import_array()


cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t* size) except NULL


ctypedef struct StringChunk:
    const uint8_t* validity  # NULL when the chunk has no nulls
    const int32_t* offsets32  # NULL for large_string
    const int64_t* offsets64  # NULL for string
    const char* data
    int64_t offset
    int64_t length


cdef inline const void* _buffer_address(object buf):
    if buf is None:
        return NULL
    if not buf.is_cpu:
        raise ValueError("arrow_regex kernels require CPU buffers")
    return <const void*><uintptr_t>buf.address


cdef StringChunk _unpack(object chunk) except *:
    cdef StringChunk c
    type_id = str(chunk.type)
    if type_id not in ("string", "large_string"):
        raise TypeError(f"expected a string or large_string array, got {type_id}")
    validity_buf, offsets_buf, data_buf = chunk.buffers()
    c.validity = (
        <const uint8_t*>_buffer_address(validity_buf)
        if chunk.null_count > 0
        else NULL
    )
    if type_id == "string":
        c.offsets32 = <const int32_t*>_buffer_address(offsets_buf)
        c.offsets64 = NULL
    else:
        c.offsets32 = NULL
        c.offsets64 = <const int64_t*>_buffer_address(offsets_buf)
    c.data = <const char*>_buffer_address(data_buf)
    c.offset = chunk.offset
    c.length = len(chunk)
    return c


cdef inline bint _is_valid(const StringChunk* c, Py_ssize_t i) noexcept nogil:
    cdef int64_t j
    if c.validity is NULL:
        return True
    j = c.offset + i
    return (c.validity[j >> 3] >> (j & 7)) & 1


cdef inline void _bounds(
    const StringChunk* c, Py_ssize_t i, int64_t* start, int64_t* stop
) noexcept nogil:
    cdef int64_t j = c.offset + i
    if c.offsets32 is not NULL:
        start[0] = c.offsets32[j]
        stop[0] = c.offsets32[j + 1]
    else:
        start[0] = c.offsets64[j]
        stop[0] = c.offsets64[j + 1]


cdef inline str _get(const StringChunk* c, Py_ssize_t i):
    cdef int64_t start, stop
    _bounds(c, i, &start, &stop)
    if start == stop:
        return ""
    return PyUnicode_DecodeUTF8(c.data + start, stop - start, NULL)


cdef object _pack_bits(cnp.ndarray[uint8_t, ndim=1] values):
    return np.packbits(values, bitorder="little")


cdef object _input_validity(const StringChunk* c):
    cdef:
        Py_ssize_t i
        cnp.ndarray[uint8_t, ndim=1] valid
    if c.validity is NULL:
        return None
    valid = np.empty(c.length, dtype=np.uint8)
    for i in range(c.length):
        valid[i] = _is_valid(c, i)
    return _pack_bits(valid)


cdef class _StringBuilder:
    """
    Accumulate strings into Arrow offsets, data and validity buffers.
    """
    cdef:
        cnp.ndarray data
        cnp.ndarray offsets
        cnp.ndarray valid
        uint8_t* data_ptr
        int64_t* offsets_ptr
        uint8_t* valid_ptr
        int64_t nbytes
        Py_ssize_t length
        Py_ssize_t capacity
        Py_ssize_t null_count

    def __cinit__(self, Py_ssize_t capacity, int64_t nbytes_hint):
        capacity = max(capacity, 1)
        self.data = np.empty(max(nbytes_hint, 16), dtype=np.uint8)
        self.offsets = np.empty(capacity + 1, dtype=np.int64)
        self.valid = np.empty(capacity, dtype=np.uint8)
        self.data_ptr = <uint8_t*>cnp.PyArray_DATA(self.data)
        self.offsets_ptr = <int64_t*>cnp.PyArray_DATA(self.offsets)
        self.valid_ptr = <uint8_t*>cnp.PyArray_DATA(self.valid)
        self.offsets_ptr[0] = 0
        self.nbytes = 0
        self.length = 0
        self.capacity = capacity
        self.null_count = 0

    cdef int _reserve_slot(self) except -1:
        cdef Py_ssize_t new_capacity
        if self.length < self.capacity:
            return 0
        new_capacity = 2 * self.capacity
        offsets = np.empty(new_capacity + 1, dtype=np.int64)
        offsets[: self.length + 1] = self.offsets[: self.length + 1]
        valid = np.empty(new_capacity, dtype=np.uint8)
        valid[: self.length] = self.valid[: self.length]
        self.offsets = offsets
        self.valid = valid
        self.offsets_ptr = <int64_t*>cnp.PyArray_DATA(self.offsets)
        self.valid_ptr = <uint8_t*>cnp.PyArray_DATA(self.valid)
        self.capacity = new_capacity
        return 0

    cdef int _reserve_bytes(self, int64_t n) except -1:
        cdef int64_t size = cnp.PyArray_SIZE(self.data)
        if self.nbytes + n <= size:
            return 0
        size = max(2 * size, self.nbytes + n)
        data = np.empty(size, dtype=np.uint8)
        data[: self.nbytes] = self.data[: self.nbytes]
        self.data = data
        self.data_ptr = <uint8_t*>cnp.PyArray_DATA(self.data)
        return 0

    cdef int append_bytes(self, const char* buf, int64_t n) except -1:
        self._reserve_slot()
        if n > 0:
            self._reserve_bytes(n)
            memcpy(self.data_ptr + self.nbytes, buf, n)
            self.nbytes += n
        self.valid_ptr[self.length] = 1
        self.length += 1
        self.offsets_ptr[self.length] = self.nbytes
        return 0

    cdef int append_str(self, object value) except -1:
        cdef:
            const char* buf
            Py_ssize_t n
        if type(value) is not str:
            if not isinstance(value, str):
                raise TypeError(
                    f"expected a str result, got {type(value).__name__}"
                )
        buf = PyUnicode_AsUTF8AndSize(value, &n)
        return self.append_bytes(buf, n)

    cdef int append_null(self) except -1:
        self._reserve_slot()
        self.valid_ptr[self.length] = 0
        self.length += 1
        self.offsets_ptr[self.length] = self.nbytes
        self.null_count += 1
        return 0

    cdef tuple finish(self, bint large):
        """
        Return (validity, offsets, data); offsets are int32 unless `large`.
        """
        offsets = self.offsets[: self.length + 1]
        if not large:
            if self.nbytes > INT32_MAX:
                raise OverflowError(
                    "result does not fit in a string array; use large_string"
                )
            offsets = offsets.astype(np.int32)
        validity = None
        if self.null_count > 0:
            validity = _pack_bits(self.valid[: self.length])
        return validity, offsets, self.data[: self.nbytes]


cdef int64_t _data_nbytes(const StringChunk* c) noexcept nogil:
    cdef int64_t start, stop, ignore
    if c.length == 0:
        return 0
    _bounds(c, 0, &start, &ignore)
    _bounds(c, c.length - 1, &ignore, &stop)
    return stop - start


def search(object chunk, object func):
    """
    Evaluate ``func(value) is not None`` for each value.

    Parameters
    ----------
    chunk : pyarrow.Array
        Array of type string or large_string.
    func : callable
        Typically the ``search``, ``match`` or ``fullmatch`` method of a
        compiled pattern.

    Returns
    -------
    validity : ndarray[uint8] or None
        Packed validity bitmap of the result; that of `chunk`, offset to 0.
    values : ndarray[uint8]
        Packed bitmap of the boolean results. Bits of null values are 0.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i
        cnp.ndarray[uint8_t, ndim=1] values = np.zeros(c.length, dtype=np.uint8)

    for i in range(c.length):
        if _is_valid(&c, i):
            values[i] = func(_get(&c, i)) is not None
    return _input_validity(&c), _pack_bits(values)


def count(object chunk, object func):
    """
    Evaluate ``len(func(value))`` for each value.

    Parameters
    ----------
    chunk : pyarrow.Array
        Array of type string or large_string.
    func : callable
        Typically the ``findall`` method of a compiled pattern.

    Returns
    -------
    validity : ndarray[uint8] or None
        Packed validity bitmap of the result; that of `chunk`, offset to 0.
    values : ndarray[int64]
        The counts. Values of null entries are 0.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i
        cnp.ndarray[int64_t, ndim=1] values = np.zeros(c.length, dtype=np.int64)

    for i in range(c.length):
        if _is_valid(&c, i):
            values[i] = len(func(_get(&c, i)))
    return _input_validity(&c), values


def map_str(object chunk, object func, bint large):
    """
    Evaluate ``func(value)`` for each value, where `func` returns a str or None.

    A result that is the input string itself is copied from the input buffer
    rather than re-encoded.

    Parameters
    ----------
    chunk : pyarrow.Array
        Array of type string or large_string.
    func : callable
        E.g. ``functools.partial(compiled.sub, repl, count=n)``.
    large : bool
        Whether to return int64 offsets (large_string) instead of int32.

    Returns
    -------
    validity : ndarray[uint8] or None
        Packed validity bitmap; null where `chunk` is null or `func` returned None.
    offsets : ndarray[int32] or ndarray[int64]
    data : ndarray[uint8]
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i
        int64_t start, stop
        str value
        _StringBuilder builder = _StringBuilder(c.length, _data_nbytes(&c))

    for i in range(c.length):
        if not _is_valid(&c, i):
            builder.append_null()
            continue
        value = _get(&c, i)
        result = func(value)
        if result is None:
            builder.append_null()
        elif result is value:
            _bounds(&c, i, &start, &stop)
            builder.append_bytes(c.data + start, stop - start)
        else:
            builder.append_str(result)
    return builder.finish(large)


def extract(object chunk, object func, Py_ssize_t ngroups, bint large):
    """
    Collect the groups of ``func(value)`` for each value.

    Parameters
    ----------
    chunk : pyarrow.Array
        Array of type string or large_string.
    func : callable
        The ``search`` method of a compiled pattern with `ngroups` groups.
    ngroups : int
        Number of groups of the pattern.
    large : bool
        Whether to return int64 offsets (large_string) instead of int32.

    Returns
    -------
    list of (validity, offsets, data)
        One entry per group, as returned by :func:`map_str`. A group is null
        where `chunk` is null, `func` found no match, or the group did not
        participate in the match.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i, k
        int64_t nbytes = _data_nbytes(&c) // max(ngroups, 1)
        list builders = [_StringBuilder(c.length, nbytes) for _ in range(ngroups)]
        _StringBuilder builder
        tuple groups

    for i in range(c.length):
        match = func(_get(&c, i)) if _is_valid(&c, i) else None
        if match is None:
            for k in range(ngroups):
                (<_StringBuilder>builders[k]).append_null()
            continue
        groups = match.groups()
        for k in range(ngroups):
            builder = <_StringBuilder>builders[k]
            group = groups[k]
            if group is None:
                builder.append_null()
            else:
                builder.append_str(group)
    results = []
    for k in range(ngroups):
        results.append((<_StringBuilder>builders[k]).finish(large))
    return results


def map_str_list(object chunk, object func, bint large):
    """
    Evaluate ``func(value)`` for each value, where `func` returns a list whose
    elements are str or None.

    Parameters
    ----------
    chunk : pyarrow.Array
        Array of type string or large_string.
    func : callable
        E.g. the ``findall`` or ``split`` method of a compiled pattern.
    large : bool
        Whether to return int64 offsets for the strings (large_string) instead
        of int32. The list offsets are always int32.

    Returns
    -------
    validity : ndarray[uint8] or None
        Packed validity bitmap of the lists; that of `chunk`, offset to 0.
    list_offsets : ndarray[int32]
    child : tuple of (validity, offsets, data)
        The list elements, as returned by :func:`map_str`.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i
        int64_t n = 0
        cnp.ndarray[int64_t, ndim=1] list_offsets = np.empty(
            c.length + 1, dtype=np.int64
        )
        _StringBuilder child = _StringBuilder(c.length, _data_nbytes(&c))
        list items

    list_offsets[0] = 0
    for i in range(c.length):
        if _is_valid(&c, i):
            items = func(_get(&c, i))
            for item in items:
                if item is None:
                    child.append_null()
                else:
                    child.append_str(item)
            n += len(items)
        list_offsets[i + 1] = n

    if n > INT32_MAX:
        raise OverflowError("result does not fit in a list array")
    return _input_validity(&c), list_offsets.astype(np.int32), child.finish(large)
