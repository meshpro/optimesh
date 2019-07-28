# -*- coding: utf-8 -*-
#
from __future__ import print_function

from . import cli, cpt, cvt, laplace, odt
from .__about__ import (
    __author__,
    __author_email__,
    __license__,
    __status__,
    __version__,
    __website__,
)

__all__ = [
    "__version__",
    "__author__",
    "__author_email__",
    "__website__",
    "__status__",
    "__license__",
    "cli",
    "cpt",
    "cvt",
    "laplace",
    "odt",
]

# try:
#     import pipdate
# except ImportError:
#     pass
# else:
#     if pipdate.needs_checking(__name__):
#         print(pipdate.check(__name__, __version__), end='')
