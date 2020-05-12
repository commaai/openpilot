
""" This file contains some helpers methods in both Python2 and 3 """
import sys
import re

if sys.version < '3':
    # Python2.x imports
    import Queue
    import codecs
else:
    # Python 3.x imports
    import queue


def check_token(token):
    """ Checks if the given token is a valid UUID."""
    valid = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
                       r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

    return valid.match(token)

# We need to do some things different pending if its Python 2.x or 3.x
if sys.version < '3':
    def to_unicode(ch):
        return codecs.unicode_escape_decode(ch)[0]

    def is_unicode(ch):
        return isinstance(ch, unicode)

    def create_unicode(ch):
        try:
            return unicode(ch, 'utf-8')
        except UnicodeDecodeError as e:
            return str(e)

    def create_queue(max_size):
        return Queue.Queue(max_size)
else:
    def to_unicode(ch):
        return ch

    def is_unicode(ch):
        return isinstance(ch, str)

    def create_unicode(ch):
        return str(ch)

    def create_queue(max_size):
        return queue.Queue(max_size)
