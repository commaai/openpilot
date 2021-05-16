#!/bin/bash -e

# workaround for a git lfs bug when pushing
# to PR branches that don't have lfs enabled

git lfs uninstall
git push
git lfs install
