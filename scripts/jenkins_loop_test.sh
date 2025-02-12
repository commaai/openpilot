#!/usr/bin/env bash
set -e

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

COOKIE_JAR=/tmp/cookies
CRUMB=$(curl -s --cookie-jar $COOKIE_JAR 'https://jenkins.comma.life/crumbIssuer/api/xml?xpath=concat(//crumbRequestField,":",//crumb)')

TESTING_BRANCH="__jenkins_loop_test_bps"
API_ROUTE="https://jenkins.comma.life/job/openpilot/job/$TESTING_BRANCH"
N=2
TEST_BUILDS=()

# Try to find previous builds
ALL_BUILDS=( $(curl -s $API_ROUTE/api/json | jq .builds.[].number || :) )

# No builds. Create branch
if [[ ${#ALL_BUILDS[@]} -eq 0 ]]; then
  TEMP_DIR=$(mktemp -d)
  GIT_LFS_SKIP_SMUDGE=1 git clone -b enable-bps --depth=1 --no-tags git@github.com:commaai/openpilot $TEMP_DIR
  git -C $TEMP_DIR checkout -b $TESTING_BRANCH
  echo "TESTING" >> $TEMP_DIR/testing_jenkins
  cp /home/batman/openpilot/Jenkinsfile $TEMP_DIR
  git -C $TEMP_DIR add testing_jenkins
  git -C $TEMP_DIR add Jenkinsfile
  git -C $TEMP_DIR commit -m "testing"
  git -C $TEMP_DIR push -f origin $TESTING_BRANCH
  rm -rf $TEMP_DIR
  FIRST_RUN=1
  sleep 90
else
  # Found some builds. Check if they are still running
  for i in ${ALL_BUILDS[@]}; do
    running=$(curl -s $API_ROUTE/$i/api/json/ | jq .inProgress)
    if [[ $running == "false" ]]; then
      continue
    fi
    TEST_BUILDS=( ${ALL_BUILDS[@]} )
    N=${#TEST_BUILDS[@]}
    break
  done
fi

# No running builds found
if [[ ${#TEST_BUILDS[@]} -eq 0 ]]; then
  FIRST_RUN=$(curl -s $API_ROUTE/api/json | jq .nextBuildNumber)
  LAST_RUN=$((FIRST_RUN+N-1))
  TEST_BUILDS=( $(seq $FIRST_RUN $LAST_RUN) )

  # Start N new builds
  for i in ${TEST_BUILDS[@]};
  do
    echo "Starting build $i"
    curl -s --output /dev/null --cookie $COOKIE_JAR -H "$CRUMB" -X POST $API_ROUTE/build?delay=0sec
    sleep 5
  done
fi

echo "Testing Jenkins with $N builds:"

while true; do
  sleep 60

  count=0
  for i in ${TEST_BUILDS[@]};
  do
    RES=$(curl -s -w "\n%{http_code}" --cookie $COOKIE_JAR -H "$CRUMB" $API_ROUTE/$i/api/json)
    HTTP_CODE=$(tail -n1 <<< "$RES")
    JSON=$(sed '$ d' <<< "$RES")

    if [[ $HTTP_CODE == "200" ]]; then
      STILL_RUNNING=$(echo $JSON | jq .inProgress)
      if [[ $STILL_RUNNING == "true" ]]; then
        echo -e "Build $i: ${YELLOW}still running${NC}"
        continue
      else
        count=$((count+1))
        echo -e "Build $i: ${GREEN}done${NC}"
      fi
    else
      echo "No status for build $i"
    fi
  done
  echo ""

  if [[ $count -ge $N ]]; then
    break
  fi
done
