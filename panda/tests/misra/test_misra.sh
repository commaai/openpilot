#!/bin/bash -e

git clone https://github.com/danmar/cppcheck.git || true
cd cppcheck
git fetch
git checkout 44d6066c6fad32e2b0332b3f2b24bd340febaef8
make -j4
cd ../../../

# whole panda code
tests/misra/cppcheck/cppcheck --dump --enable=all --inline-suppr board/main.c 2>/tmp/misra/cppcheck_output.txt || true
python tests/misra/cppcheck/addons/misra.py board/main.c.dump 2>/tmp/misra/misra_output.txt || true

# violations in safety files
(cat /tmp/misra/misra_output.txt | grep safety) > /tmp/misra/misra_safety_output.txt || true
(cat /tmp/misra/cppcheck_output.txt | grep safety) > /tmp/misra/cppcheck_safety_output.txt || true

if [[ -s "/tmp/misra/misra_safety_output.txt" ]] || [[ -s "/tmp/misra/cppcheck_safety_output.txt" ]]
then
  echo "Found Misra violations in the safety code:"
  cat /tmp/misra/misra_safety_output.txt
  cat /tmp/misra/cppcheck_safety_output.txt
  exit 1
fi
