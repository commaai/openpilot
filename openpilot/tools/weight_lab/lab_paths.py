"""Where weight lab keeps its data (replays, thumbnails, frames). The code lives next to this file."""
import os

DATA = os.environ.get('WEIGHT_LAB_DATA', '/home/batman/xx/junk/shane/weight_edit')
CODE = os.path.dirname(os.path.abspath(__file__))
