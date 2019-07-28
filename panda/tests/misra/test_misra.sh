#!/bin/bash -e

mkdir /tmp/misra || true
git clone https://github.com/danmar/cppcheck.git || true
cd cppcheck
git fetch
git checkout 862c4ef87b109ae86c2d5f12769b7c8d199f35c5
make -j4
cd ../../../


printf "\nPANDA CODE\n"
tests/misra/cppcheck/cppcheck -DPANDA -UPEDAL -DCAN3 -DUID_BASE -DEON \
                              --suppressions-list=tests/misra/suppressions.txt \
                              --dump --enable=all --inline-suppr --force \
                              board/main.c 2>/tmp/misra/cppcheck_output.txt

python tests/misra/cppcheck/addons/misra.py board/main.c.dump 2> /tmp/misra/misra_output.txt || true

# strip (information) lines
cppcheck_output=$( cat /tmp/misra/cppcheck_output.txt | grep -v "(information) " ) || true
misra_output=$( cat /tmp/misra/misra_output.txt | grep -v "(information) " ) || true


printf "\nPEDAL CODE\n"
tests/misra/cppcheck/cppcheck -UPANDA -DPEDAL -UCAN3 \
                              --suppressions-list=tests/misra/suppressions.txt \
                              -I board/ --dump --enable=all --inline-suppr --force \
                              board/pedal/main.c 2>/tmp/misra/cppcheck_pedal_output.txt

python tests/misra/cppcheck/addons/misra.py board/pedal/main.c.dump 2> /tmp/misra/misra_pedal_output.txt || true

# strip (information) lines
cppcheck_pedal_output=$( cat /tmp/misra/cppcheck_pedal_output.txt | grep -v "(information) " ) || true
misra_pedal_output=$( cat /tmp/misra/misra_pedal_output.txt | grep -v "(information) " ) || true

if [[ -n "$misra_output" ]] || [[ -n "$cppcheck_output" ]]
then
  echo "Failed! found Misra violations in panda code:"
  echo "$misra_output"
  echo "$cppcheck_output"
  exit 1
fi

if [[ -n "$misra_pedal_output" ]] || [[ -n "$cppcheck_pedal_output" ]]
then
  echo "Failed! found Misra violations in pedal code:"
  echo "$misra_pedal_output"
  echo "$cppcheck_pedal_output"
  exit 1
fi

echo "Success"
