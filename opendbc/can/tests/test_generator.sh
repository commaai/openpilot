#!/bin/bash -e

cd ../../generator/

# run generator
./generator.py

if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
  echo "Unexpected changes after running generator.py";
  exit 1
else
  echo "Success";
fi
