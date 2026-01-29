#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

source ../../../../setup.sh

GREEN="\e[1;32m"
YELLOW="\e[1;33m"
RED="\e[1;31m"
NC='\033[0m'

: "${CPPCHECK_DIR:=$DIR/cppcheck/}"

# ensure checked in coverage table is up to date
python3 $CPPCHECK_DIR/addons/misra.py -generate-table > coverage_table
if ! git diff --quiet coverage_table; then
  echo -e "${YELLOW}MISRA coverage table doesn't match. Update and commit:${NC}"
  exit 3
fi

cd $BASEDIR

CHECKLIST=$(mktemp)
echo "Cppcheck checkers list from test_misra.sh:" > $CHECKLIST

cppcheck() {
  # get all gcc defines: arm-none-eabi-gcc -dM -E - < /dev/null
  COMMON_DEFINES="-D__GNUC__=9"

  # note that cppcheck build cache results in inconsistent results as of v2.13.0
  OUTPUT=$(mktemp)

  echo -e "\n\n\n\n\nTEST variant options:" >> $CHECKLIST
  echo -e ""${@//$BASEDIR/}"\n\n" >> $CHECKLIST # (absolute path removed)

  OPENDBC_ROOT=${OPENDBC_ROOT:-$BASEDIR}
  $CPPCHECK_DIR/cppcheck --inline-suppr -I $OPENDBC_ROOT \
          --suppress=missingIncludeSystem \
          --suppressions-list=$DIR/suppressions.txt  \
           --error-exitcode=2 --check-level=exhaustive --safety \
          --platform=arm32-wchar_t4 $COMMON_DEFINES --checkers-report=$CHECKLIST.tmp \
          --std=c11 "$@" 2>&1 | tee $OUTPUT

  cat $CHECKLIST.tmp >> $CHECKLIST
  rm $CHECKLIST.tmp
  # cppcheck bug: some MISRA errors won't result in the error exit code,
  # so check the output (https://trac.cppcheck.net/ticket/12440#no1)
  if grep -e "misra violation" -e "error" -e "style: " $OUTPUT > /dev/null; then
    printf "${RED}** FAILED: MISRA violations found!${NC}\n"
    exit 1
  fi
}

OPTS=" --enable=all --enable=unusedFunction --addon=misra"

printf "\n${GREEN}** Safety **${NC}\n"
cppcheck $OPTS $BASEDIR/opendbc/safety/tests/misra/main.c

printf "\n${GREEN}Success!${NC} took $SECONDS seconds\n"

# ensure list of checkers is up to date
if [ -z "$OPENDBC_ROOT" ]; then
  cd $DIR
  if ! git diff --quiet $CHECKLIST; then
    echo -e "\n${YELLOW}WARNING: Cppcheck checkers.txt report has changed. Review and commit...${NC}"
    mv $CHECKLIST $DIR/checkers.txt
    exit 4
  fi
fi
