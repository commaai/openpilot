import os
import logging

# set up logging
LOGPRINT = os.environ.get('LOGPRINT', 'INFO').upper()
carlog = logging.getLogger('carlog')
carlog.setLevel(LOGPRINT)
carlog.propagate = False

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
carlog.addHandler(handler)
