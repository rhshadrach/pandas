"""
Evaluate Python ``re`` patterns on the values of pyarrow string arrays.

The kernels read the validity bitmap, offsets and UTF-8 data of a ``string`` or
``large_string`` array directly from its buffers, decode each valid value to a
``str``, call a method of a compiled ``re.Pattern`` on it, and write the result
into the buffers of a new Arrow array. No intermediate list or object ndarray of
the values is created, and the results are not converted from Python objects by
pyarrow.

All kernels take a single ``pyarrow.Array`` (one chunk). String results are
returned as pyarrow arrays of the type of the chunk; other results as numpy
arrays that the caller wraps with ``pyarrow.Array.from_buffers``. Bitmaps are
packed, in Arrow's bit order. A validity bitmap is None when there are no nulls.
"""
from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_New,
)
from cpython.unicode cimport PyUnicode_DecodeUTF8
from cython cimport Py_ssize_t
from libc.stdint cimport (
    INT32_MAX,
    int32_t,
    int64_t,
    uint8_t,
    uintptr_t,
)
from libc.stdlib cimport (
    free,
    malloc,
    realloc,
)
from libc.string cimport (
    memcpy,
    memset,
)

import numpy as np

cimport numpy as cnp

from pandas.compat.pyarrow import HAS_PYARROW

cnp.import_array()

cdef object _pa_foreign_buffer = None
cdef object _pa_from_buffers = None

if HAS_PYARROW:
    import pyarrow as pa

    _pa_foreign_buffer = pa.foreign_buffer
    _pa_from_buffers = pa.Array.from_buffers


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


cdef inline void _set_bit(uint8_t* bits, Py_ssize_t i) noexcept nogil:
    bits[i >> 3] |= <uint8_t>(1 << (i & 7))


cdef inline cnp.ndarray _new_bitmap(Py_ssize_t length):
    # all bits 0
    return np.zeros((length + 7) // 8, dtype=np.uint8)


cdef object _input_validity(const StringChunk* c):
    # the validity bitmap of `c`, offset to 0
    cdef:
        Py_ssize_t i
        cnp.ndarray bitmap
        uint8_t* bits
    if c.validity is NULL:
        return None
    bitmap = _new_bitmap(c.length)
    bits = <uint8_t*>cnp.PyArray_DATA(bitmap)
    for i in range(c.length):
        if _is_valid(c, i):
            _set_bit(bits, i)
    return bitmap


cdef void _free_malloc_capsule(object capsule) noexcept:
    free(PyCapsule_GetPointer(capsule, NULL))


cdef object _foreign_buffer(void* ptr, int64_t size):
    # hand a malloc'd buffer to pyarrow, which frees it when no longer used
    capsule = PyCapsule_New(ptr, NULL, _free_malloc_capsule)
    return _pa_foreign_buffer(<uintptr_t>ptr, size, capsule)


cdef class _StringBuilder:
    """
    Accumulate strings into the offsets, data and validity buffers of an Arrow
    ``string`` or ``large_string`` array.

    The buffers are malloc'd and move to pyarrow in `finish`; whatever has not
    moved is freed with the builder, e.g. when the kernel filling it raises.
    """
    cdef:
        int32_t* offsets32  # NULL for large_string
        int64_t* offsets64  # NULL for string
        uint8_t* validity
        char* data
        int64_t nbytes
        int64_t data_capacity
        Py_ssize_t length
        Py_ssize_t capacity
        Py_ssize_t null_count
        bint large
        object pa_type

    def __cinit__(self, object pa_type, Py_ssize_t capacity, int64_t nbytes_hint):
        self.pa_type = pa_type
        self.large = str(pa_type) == "large_string"
        self.capacity = max(capacity, 1)
        self.data_capacity = max(nbytes_hint, 16)
        if self.large:
            self.offsets64 = <int64_t*>malloc((self.capacity + 1) * sizeof(int64_t))
            if self.offsets64 is NULL:
                raise MemoryError()
            self.offsets64[0] = 0
        else:
            self.offsets32 = <int32_t*>malloc((self.capacity + 1) * sizeof(int32_t))
            if self.offsets32 is NULL:
                raise MemoryError()
            self.offsets32[0] = 0
        # all valid until a null clears its bit
        self.validity = <uint8_t*>malloc((self.capacity + 7) // 8)
        self.data = <char*>malloc(self.data_capacity)
        if self.validity is NULL or self.data is NULL:
            raise MemoryError()
        memset(self.validity, 0xFF, (self.capacity + 7) // 8)

    def __dealloc__(self):
        free(self.offsets32)
        free(self.offsets64)
        free(self.validity)
        free(self.data)

    cdef int _reserve_slot(self) except -1:
        cdef:
            Py_ssize_t capacity = 2 * self.capacity
            Py_ssize_t nbytes = (self.capacity + 7) // 8
            Py_ssize_t new_nbytes = (capacity + 7) // 8
            void* grown
        if self.length < self.capacity:
            return 0
        if self.large:
            grown = realloc(self.offsets64, (capacity + 1) * sizeof(int64_t))
            if grown is NULL:
                raise MemoryError()
            self.offsets64 = <int64_t*>grown
        else:
            grown = realloc(self.offsets32, (capacity + 1) * sizeof(int32_t))
            if grown is NULL:
                raise MemoryError()
            self.offsets32 = <int32_t*>grown
        grown = realloc(self.validity, new_nbytes)
        if grown is NULL:
            raise MemoryError()
        self.validity = <uint8_t*>grown
        memset(self.validity + nbytes, 0xFF, new_nbytes - nbytes)
        self.capacity = capacity
        return 0

    cdef int _reserve_bytes(self, int64_t n) except -1:
        cdef:
            int64_t capacity
            void* grown
        if self.nbytes + n <= self.data_capacity:
            return 0
        if not self.large and self.nbytes + n > INT32_MAX:
            raise OverflowError(
                "result does not fit in a string array; use large_string"
            )
        capacity = max(2 * self.data_capacity, self.nbytes + n)
        if not self.large:
            capacity = min(capacity, <int64_t>INT32_MAX)
        grown = realloc(self.data, capacity)
        if grown is NULL:
            raise MemoryError()
        self.data = <char*>grown
        self.data_capacity = capacity
        return 0

    cdef inline void _end_value(self) noexcept:
        self.length += 1
        if self.large:
            self.offsets64[self.length] = self.nbytes
        else:
            self.offsets32[self.length] = <int32_t>self.nbytes

    cdef int append_bytes(self, const char* buf, int64_t n) except -1:
        self._reserve_slot()
        if n > 0:
            self._reserve_bytes(n)
            memcpy(self.data + self.nbytes, buf, n)
            self.nbytes += n
        self._end_value()
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
        self.validity[self.length >> 3] &= <uint8_t>~(1 << (self.length & 7))
        self.null_count += 1
        self._end_value()
        return 0

    cdef object finish(self):
        """
        Move the buffers to a pyarrow array; can be called only once.
        """
        cdef void* ptr
        if self.data is NULL:
            raise RuntimeError("finish() may only be called once")

        if self.large:
            ptr = self.offsets64
            self.offsets64 = NULL
            offsets = _foreign_buffer(ptr, (self.length + 1) * sizeof(int64_t))
        else:
            ptr = self.offsets32
            self.offsets32 = NULL
            offsets = _foreign_buffer(ptr, (self.length + 1) * sizeof(int32_t))

        validity = None
        if self.null_count > 0:
            ptr = self.validity
            self.validity = NULL
            validity = _foreign_buffer(ptr, (self.length + 7) // 8)

        ptr = self.data
        self.data = NULL
        data = _foreign_buffer(ptr, self.nbytes)

        return _pa_from_buffers(
            self.pa_type,
            self.length,
            [validity, offsets, data],
            null_count=self.null_count,
        )


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
        cnp.ndarray values = _new_bitmap(c.length)
        uint8_t* bits = <uint8_t*>cnp.PyArray_DATA(values)

    for i in range(c.length):
        if _is_valid(&c, i) and func(_get(&c, i)) is not None:
            _set_bit(bits, i)
    return _input_validity(&c), values


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


def map_str(object chunk, object func):
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

    Returns
    -------
    pyarrow.Array
        Of the type of `chunk`; null where `chunk` is null or `func` returned
        None.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i
        int64_t start, stop
        str value
        _StringBuilder builder = _StringBuilder(chunk.type, c.length, _data_nbytes(&c))

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
    return builder.finish()


def extract(object chunk, object func, Py_ssize_t ngroups):
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

    Returns
    -------
    list of pyarrow.Array
        One array per group, of the type of `chunk`. A group is null where
        `chunk` is null, `func` found no match, or the group did not
        participate in the match.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i, k
        int64_t nbytes = _data_nbytes(&c) // max(ngroups, 1)
        list builders = [
            _StringBuilder(chunk.type, c.length, nbytes) for _ in range(ngroups)
        ]
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
        results.append((<_StringBuilder>builders[k]).finish())
    return results


def map_str_list(object chunk, object func):
    """
    Evaluate ``func(value)`` for each value, where `func` returns a list whose
    elements are str or None.

    Parameters
    ----------
    chunk : pyarrow.Array
        Array of type string or large_string.
    func : callable
        E.g. the ``findall`` or ``split`` method of a compiled pattern.

    Returns
    -------
    validity : ndarray[uint8] or None
        Packed validity bitmap of the lists; that of `chunk`, offset to 0.
    list_offsets : ndarray[int32]
    child : pyarrow.Array
        The list elements, of the type of `chunk`.
    """
    cdef:
        StringChunk c = _unpack(chunk)
        Py_ssize_t i
        int64_t n = 0
        cnp.ndarray[int32_t, ndim=1] list_offsets = np.empty(
            c.length + 1, dtype=np.int32
        )
        _StringBuilder child = _StringBuilder(chunk.type, c.length, _data_nbytes(&c))
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
            if n > INT32_MAX:
                raise OverflowError("result does not fit in a list array")
        list_offsets[i + 1] = <int32_t>n
    return _input_validity(&c), list_offsets, child.finish()
