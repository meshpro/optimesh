# -*- coding: utf-8 -*-
#
from __future__ import print_function


try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end='')
