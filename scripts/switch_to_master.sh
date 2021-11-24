#!/usr/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $DIR/..

git clean -xdf .
git rm -r --cached .

git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git fetch origin master
git checkout master
git reset --hard
git submodule update --init

printf '\n\n'
echo "master checked out. reboot to start building openpilot master"
