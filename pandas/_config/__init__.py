"""
pandas._config is considered explicitly upstream of everything else in pandas,
should have no intra-pandas dependencies.

importing `dates` and `display` ensures that keys needed by _libs
are initialized.
"""

__all__ = [
    "describe_option",
    "detect_console_encoding",
    "get_option",
    "option_context",
    "options",
    "reset_option",
    "set_option",
]
from pandas._config import dates  # pyright: ignore[reportUnusedImport]  # noqa: F401
from pandas._config.config import (
    describe_option,
    get_option,
    option_context,
    options,
    reset_option,
    set_option,
)
from pandas._config.display import detect_console_encoding


def using_string_dtype() -> bool:
    from pandas._config.config import _global_config as config

    _mode_options = config["future"]
    return _mode_options["infer_string"]


# Performance: using_python_scalars is called on hot scalar-access paths
# (e.g. Series._ixs, Series._get_value), and a function-level import costs
# roughly ten times a dict lookup. The "future" sub-dict of _global_config is
# created once at option registration and only ever mutated in place, so the
# reference can be cached.
_future_options: dict | None = None


def using_python_scalars() -> bool:
    global _future_options
    if _future_options is None:
        from pandas._config.config import _global_config

        _future_options = _global_config["future"]
    return _future_options["python_scalars"]


def is_nan_na() -> bool:
    from pandas._config.config import _global_config as config

    _mode_options = config["future"]
    return not _mode_options["distinguish_nan_and_na"]


def using_infer_freq_offset() -> bool | None:
    from pandas._config.config import _global_config as config

    _mode_options = config["future"]
    return _mode_options["infer_freq_returns_offset"]
