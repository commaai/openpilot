from .manager import JSONRPCResponseManager
from .dispatcher import Dispatcher

__version = (1, 12, 1)

__version__ = version = '.'.join(map(str, __version))
__project__ = PROJECT = __name__

dispatcher = Dispatcher()

# lint_ignore=W0611,W0401
