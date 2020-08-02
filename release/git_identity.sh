#!/usr/bin/env bash

export GIT_COMMITTER_NAME="Vehicle Researcher"
export GIT_COMMITTER_EMAIL="user@comma.ai"
export GIT_AUTHOR_NAME="Vehicle Researcher"
export GIT_AUTHOR_EMAIL="user@comma.ai"

if [ -f /data/gitkey ]; then
  export GIT_SSH_COMMAND="ssh -i /data/gitkey"
fi
