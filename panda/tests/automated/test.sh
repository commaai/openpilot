#!/bin/bash -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
$(which nosetests) -v -s $(ls $DIR/$1*.py)
