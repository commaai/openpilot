#!/bin/bash -e

PANDA_DIR=../..

mkdir /tmp/misra || true

# generate coverage matrix
#python tests/misra/cppcheck/addons/misra.py -generate-table > tests/misra/coverage_table

printf "\nPANDA CODE\n"
cppcheck -DPANDA -UPEDAL -DCAN3 -DUID_BASE -DEON \
         --suppressions-list=suppressions.txt \
         --dump --enable=all --inline-suppr --force \
         $PANDA_DIR/board/main.c 2>/tmp/misra/cppcheck_output.txt

python /usr/share/cppcheck/addons/misra.py $PANDA_DIR/board/main.c.dump 2> /tmp/misra/misra_output.txt || true

# strip (information) lines
cppcheck_output=$( cat /tmp/misra/cppcheck_output.txt | grep -v ": information: " ) || true
misra_output=$( cat /tmp/misra/misra_output.txt | grep -v ": information: " ) || true


printf "\nPEDAL CODE\n"
cppcheck -UPANDA -DPEDAL -UCAN3 \
         --suppressions-list=suppressions.txt \
         -I $PANDA_DIR/board/ --dump --enable=all --inline-suppr --force \
         $PANDA_DIR/board/pedal/main.c 2>/tmp/misra/cppcheck_pedal_output.txt

python /usr/share/cppcheck/addons/misra.py $PANDA_DIR/board/pedal/main.c.dump 2> /tmp/misra/misra_pedal_output.txt || true

# strip (information) lines
cppcheck_pedal_output=$( cat /tmp/misra/cppcheck_pedal_output.txt | grep -v ": information: " ) || true
misra_pedal_output=$( cat /tmp/misra/misra_pedal_output.txt | grep -v ": information: " ) || true

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
