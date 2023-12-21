#!/bin/bash
set -e

cd /tmp/openpilot.git

git log --pretty=format:'%T %h' | while read tree commit; do git ls-tree -r $tree | grep $1 && echo $commit; done
