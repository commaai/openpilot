#!/bin/bash -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd "$DIR"
sudo -- bash -c "source /etc/profile.d/comma_dev.sh; pip install pip==20.1.1 pipenv==2020.8.13; pipenv install --dev --deploy --system"
