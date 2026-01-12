try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import OMEZarrpariWidget

__all__ = ["OMEZarrpariWidget"]
