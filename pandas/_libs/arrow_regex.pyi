from collections.abc import Callable
from typing import (
    Any,
    TypeAlias,
)

import numpy as np
import pyarrow as pa

from pandas._typing import npt

_Bitmap: TypeAlias = npt.NDArray[np.uint8] | None
_StringBuffers: TypeAlias = tuple[_Bitmap, np.ndarray, npt.NDArray[np.uint8]]

def search(
    chunk: pa.Array, func: Callable[[str], Any]
) -> tuple[_Bitmap, npt.NDArray[np.uint8]]: ...
def count(
    chunk: pa.Array, func: Callable[[str], Any]
) -> tuple[_Bitmap, npt.NDArray[np.int64]]: ...
def map_str(
    chunk: pa.Array, func: Callable[[str], str | None], large: bool
) -> _StringBuffers: ...
def extract(
    chunk: pa.Array, func: Callable[[str], Any], ngroups: int, large: bool
) -> list[_StringBuffers]: ...
def map_str_list(
    chunk: pa.Array, func: Callable[[str], list[str | None]], large: bool
) -> tuple[_Bitmap, npt.NDArray[np.int32], _StringBuffers]: ...
