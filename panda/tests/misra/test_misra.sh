#!/bin/bash -e

PANDA_DIR=../..

mkdir /tmp/misra || true
ERROR_CODE=0

# generate coverage matrix
#python tests/misra/cppcheck/addons/misra.py -generate-table > tests/misra/coverage_table

printf "\nPANDA F4 CODE\n"
cppcheck -DPANDA -DSTM32F4 -UPEDAL -DCAN3 -DUID_BASE \
         --suppressions-list=suppressions.txt --suppress=*:*inc/* \
         -I $PANDA_DIR/board/ --dump --enable=all --inline-suppr --force \
         $PANDA_DIR/board/main.c 2>/tmp/misra/cppcheck_f4_output.txt

python /usr/share/cppcheck/addons/misra.py $PANDA_DIR/board/main.c.dump 2> /tmp/misra/misra_f4_output.txt || true

# strip (information) lines
cppcheck_f4_output=$( cat /tmp/misra/cppcheck_f4_output.txt | grep -v ": information: " ) || true
misra_f4_output=$( cat /tmp/misra/misra_f4_output.txt | grep -v ": information: " ) || true


printf "\nPANDA H7 CODE\n"
cppcheck -DPANDA -DSTM32H7 -UPEDAL -DUID_BASE \
         --suppressions-list=suppressions.txt --suppress=*:*inc/* \
         -I $PANDA_DIR/board/ --dump --enable=all --inline-suppr --force \
         $PANDA_DIR/board/main.c 2>/tmp/misra/cppcheck_h7_output.txt

python /usr/share/cppcheck/addons/misra.py $PANDA_DIR/board/main.c.dump 2> /tmp/misra/misra_h7_output.txt || true

# strip (information) lines
cppcheck_h7_output=$( cat /tmp/misra/cppcheck_h7_output.txt | grep -v ": information: " ) || true
misra_h7_output=$( cat /tmp/misra/misra_h7_output.txt | grep -v ": information: " ) || true


printf "\nPEDAL CODE\n"
cppcheck -UPANDA -DSTM32F2 -DPEDAL -UCAN3 \
         --suppressions-list=suppressions.txt --suppress=*:*inc/* \
         -I $PANDA_DIR/board/ --dump --enable=all --inline-suppr --force \
         $PANDA_DIR/board/pedal/main.c 2>/tmp/misra/cppcheck_pedal_output.txt

python /usr/share/cppcheck/addons/misra.py $PANDA_DIR/board/pedal/main.c.dump 2> /tmp/misra/misra_pedal_output.txt || true

# strip (information) lines
cppcheck_pedal_output=$( cat /tmp/misra/cppcheck_pedal_output.txt | grep -v ": information: " ) || true
misra_pedal_output=$( cat /tmp/misra/misra_pedal_output.txt | grep -v ": information: " ) || true

if [[ -n "$misra_f4_output" ]] || [[ -n "$cppcheck_f4_output" ]]
then
  echo "Failed! found Misra violations in panda F4 code:"
  echo "$misra_f4_output"
  echo "$cppcheck_f4_output"
  ERROR_CODE=1
fi

if [[ -n "$misra_h7_output" ]] || [[ -n "$cppcheck_h7_output" ]]
then
  echo "Failed! found Misra violations in panda H7 code:"
  echo "$misra_h7_output"
  echo "$cppcheck_h7_output"
  ERROR_CODE=1
fi

if [[ -n "$misra_pedal_output" ]] || [[ -n "$cppcheck_pedal_output" ]]
then
  echo "Failed! found Misra violations in pedal code:"
  echo "$misra_pedal_output"
  echo "$cppcheck_pedal_output"
  ERROR_CODE=1
fi

if [[ $ERROR_CODE > 0 ]]
then
  exit 1
fi

echo "Success"
