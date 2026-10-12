"""Select native RUDA or direct NVIDIA math-library execution per caller."""
from contextlib import contextmanager
from contextvars import ContextVar
import os

_default = os.environ.get('RUDA_TORCH_MATH_BACKEND', 'ruda').lower()
if _default not in ('ruda', 'nvidia'):
    raise ValueError('RUDA_TORCH_MATH_BACKEND must be ruda or nvidia')
_backend = ContextVar('ruda_math_backend', default=_default)


def get_math_backend():
    return _backend.get()


def set_math_backend(name):
    if name not in ('ruda', 'nvidia'):
        raise ValueError('math backend must be ruda or nvidia')
    previous = _backend.get()
    _backend.set(name)
    return previous


@contextmanager
def math_backend(name):
    if name not in ('ruda', 'nvidia'):
        raise ValueError('math backend must be ruda or nvidia')
    token = _backend.set(name)
    try:
        yield
    finally:
        _backend.reset(token)


def _nvidia():
    return _backend.get() == 'nvidia'
