import os
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
travis = gh_actions = BASEDIR.strip('/').split('/')[0] != 'data'
