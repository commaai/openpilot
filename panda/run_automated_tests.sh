#!/bin/bash -e
TEST_FILENAME=${TEST_FILENAME:-nosetests.xml}
if [ -f "/EON" ]; then
  TESTSUITE_NAME="Panda_Test-EON"
else
  TESTSUITE_NAME="Panda_Test-DEV"
fi

TEST_SCRIPTS=$(ls tests/automated/$1*.py)

IFS=$'\n'
for NAME in $(nmcli --fields NAME con show | grep panda | awk '{$1=$1};1')
do
  nmcli connection delete "$NAME"
done

PYTHONPATH="." $(which nosetests) -v --with-xunit --xunit-file=./$TEST_FILENAME --xunit-testsuite-name=$TESTSUITE_NAME -s $TEST_SCRIPTS
