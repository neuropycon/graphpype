from . import pipelines  # noqa
from . import interfaces  # noqa

__version__ = "unknown"
try:
    from ._version import __version__ # noqa
except ImportError:
    # We're running in a tree that doesn't have a _version.py
    pass
