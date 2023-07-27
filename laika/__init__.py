import logging
import os

from .astro_dog import AstroDog
assert AstroDog is not None

# setup logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL, format='%(message)s')
