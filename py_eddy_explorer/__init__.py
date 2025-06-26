"""
"""

from ._version import get_versions
from .contour import EddyContour  # noqa: F401

__version__ = get_versions()["version"]
del get_versions
