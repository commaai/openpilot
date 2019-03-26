"""
websocket - WebSocket client library for Python

Copyright (C) 2010 Hiroki Ohtani(liris)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA  02110-1335  USA

"""
import logging

_logger = logging.getLogger('websocket')
try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

_logger.addHandler(NullHandler())

_traceEnabled = False

__all__ = ["enableTrace", "dump", "error", "warning", "debug", "trace",
           "isEnabledForError", "isEnabledForDebug"]


def enableTrace(traceable, handler = logging.StreamHandler()):
    """
    turn on/off the traceability.

    traceable: boolean value. if set True, traceability is enabled.
    """
    global _traceEnabled
    _traceEnabled = traceable
    if traceable:
        _logger.addHandler(handler)
        _logger.setLevel(logging.DEBUG)


def dump(title, message):
    if _traceEnabled:
        _logger.debug("--- " + title + " ---")
        _logger.debug(message)
        _logger.debug("-----------------------")


def error(msg):
    _logger.error(msg)


def warning(msg):
    _logger.warning(msg)


def debug(msg):
    _logger.debug(msg)


def trace(msg):
    if _traceEnabled:
        _logger.debug(msg)


def isEnabledForError():
    return _logger.isEnabledFor(logging.ERROR)


def isEnabledForDebug():
    return _logger.isEnabledFor(logging.DEBUG)
