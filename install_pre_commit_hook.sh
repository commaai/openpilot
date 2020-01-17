#!/usr/bin/env sh
if [ -z $(which flake8) ]; then
  echo "Installing flake8"
  sudo pip install flake8
fi

echo "Setting up commit hook"
cp -u --remove-destination pre-commit.sh .git/hooks/pre-commit
