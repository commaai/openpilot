# -*- coding: utf-8 -
#
# This file is part of gunicorn released under the MIT license.
# See the NOTICE for more information.

import sys

from gunicorn import util

if sys.version_info >= (3, 4):
    try:
        import aiohttp  # pylint: disable=unused-import
    except ImportError:
        raise RuntimeError("You need aiohttp installed to use this worker.")
    else:
        try:
            from aiohttp.worker import GunicornWebWorker as AiohttpWorker
        except ImportError:
            from gunicorn.workers._gaiohttp import AiohttpWorker

        util.warn(
            "The 'gaiohttp' worker is deprecated. See --worker-class "
            "documentation for more information."
        )
        __all__ = ['AiohttpWorker']
else:
    raise RuntimeError("You need Python >= 3.4 to use the gaiohttp worker")
