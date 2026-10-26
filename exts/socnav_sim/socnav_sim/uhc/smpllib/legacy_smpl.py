"""Read legacy SMPL pickles without exposing Chumpy objects to model code."""

from collections import namedtuple
from contextlib import contextmanager
import inspect
from pathlib import Path
import pickle
from threading import RLock
from types import SimpleNamespace

import numpy as np

_LOAD_LOCK = RLock()


@contextmanager
def _legacy_chumpy_api():
    """Supply removed APIs only while importing/unpickling the legacy model."""
    aliases = {"bool": bool, "int": int, "float": float, "complex": complex,
               "object": object, "str": str, "unicode": str}
    argspec_type = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(function):
        spec = inspect.getfullargspec(function)
        return argspec_type(spec.args, spec.varargs, spec.varkw, spec.defaults)

    with _LOAD_LOCK:
        missing = {name: value for name, value in aliases.items() if name not in vars(np)}
        needs_argspec = "getargspec" not in vars(inspect)
        try:
            for name, value in missing.items():
                setattr(np, name, value)
            if needs_argspec:
                inspect.getargspec = getargspec
            yield
        finally:
            for name in missing:
                delattr(np, name)
            if needs_argspec:
                delattr(inspect, "getargspec")


def load_smpl_data(model_path, gender="neutral"):
    """Return SMPL data with Chumpy values materialized as NumPy arrays."""
    path = Path(model_path)
    if path.is_dir():
        path = path / f"SMPL_{gender.upper()}.pkl"
    with _legacy_chumpy_api():
        with path.open("rb") as stream:
            data = pickle.load(stream, encoding="latin1")
        # Preserve sparse matrices and metadata; only Chumpy values need conversion.
        data = {key: np.array(value.r, copy=True)
                if type(value).__module__.split(".")[0] == "chumpy" else value
                for key, value in data.items()}
    return SimpleNamespace(**data)
