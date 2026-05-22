from collections import namedtuple


NV12Frame = namedtuple("NV12Frame", ['width', 'height', 'stride', 'y_height', 'uv_height', 'size'])
