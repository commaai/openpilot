#!/bin/bash -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$DIR"
sudo -- bash -c "source /etc/profile.d/comma_dev.sh; pip install pip==20.1.1 git+git://github.com/pypa/pipenv.git@7a12dbb5cacc71d1dd2d74d8cce8eb50ce2db121; pipenv install --dev --deploy --system"
