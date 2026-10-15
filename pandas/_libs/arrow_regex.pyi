from collections.abc import Callable
from typing import (
    Any,
    TypeAlias,
)

import numpy as np
import pyarrow as pa

from pandas._typing import npt

_Bitmap: TypeAlias = npt.NDArray[np.uint8] | None

def search(
    chunk: pa.Array, func: Callable[[str], Any]
) -> tuple[_Bitmap, npt.NDArray[np.uint8]]: ...
def count(
    chunk: pa.Array, func: Callable[[str], Any]
) -> tuple[_Bitmap, npt.NDArray[np.int64]]: ...
def map_str(chunk: pa.Array, func: Callable[[str], str | None]) -> pa.Array: ...
def extract(
    chunk: pa.Array, func: Callable[[str], Any], ngroups: int
) -> list[pa.Array]: ...
def map_str_list(
    chunk: pa.Array, func: Callable[[str], list[str | None]]
) -> tuple[_Bitmap, npt.NDArray[np.int32], pa.Array]: ...
