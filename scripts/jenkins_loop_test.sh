#!/usr/bin/env bash
set -e

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

COOKIE_JAR=/tmp/cookies
CRUMB=$(curl -s --cookie-jar $COOKIE_JAR 'https://jenkins.comma.life/crumbIssuer/api/xml?xpath=concat(//crumbRequestField,":",//crumb)')

TESTING_BRANCH="__jenkins_loop_test"
API_ROUTE="https://jenkins.comma.life/job/openpilot/job/$TESTING_BRANCH"
N=20
TEST_BUILDS=()

# Try to find previous builds
ALL_BUILDS=( $(curl -s $API_ROUTE/api/json | jq .builds.[].number || :) )

# No builds. Create branch
if [[ ${#ALL_BUILDS[@]} -eq 0 ]]; then
  TEMP_DIR=$(mktemp -d)
  GIT_LFS_SKIP_SMUDGE=1 git clone -b master --depth=1 --no-tags git@github.com:commaai/openpilot $TEMP_DIR
  git -C $TEMP_DIR checkout -b $TESTING_BRANCH
  echo "TESTING" >> $TEMP_DIR/testing_jenkins
  git -C $TEMP_DIR add testing_jenkins
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
  # Delete all previous builds
  for i in ${ALL_BUILDS[@]}; do
    curl -s --cookie $COOKIE_JAR -H "$CRUMB" -X POST $API_ROUTE/$i/doDelete
  done

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
  #sleep 60
  sleep 5

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

STAGES_NAMES=()
while read stage; do
  STAGES_NAMES[$index]=$stage
  index=$((index+1))
done < <(curl -s -H "$CRUMB" $API_ROUTE/lastBuild/wfapi/ | jq .stages[].name)
STAGES_COUNT=${#STAGES_NAMES[@]}

STAGES_FAILURES=($(for i in $(seq 1 $STAGES_COUNT); do echo 0; done))

for i in ${TEST_BUILDS[@]}; do
index=0
while read result; do
  if [[ $result != '"SUCCESS"' ]]; then
    STAGES_FAILURES[$index]=$((STAGES_FAILURES[$index]+1))
  fi
  index=$((index+1))
done < <(curl -s $API_ROUTE/$i/wfapi/ | jq .stages[].status)
done

echo "=========================================="
echo "===================DONE==================="
echo "=========================================="

# print results of all builds
for i in $(seq 0 $(($STAGES_COUNT-1))); do
  P=$((${STAGES_FAILURES[$i]}*100/$N))
  if [[ $P -eq 0 ]]; then
    P=${GREEN}${BOLD}${P}
  else
    P=${RED}${BOLD}${P}
  fi
  echo -e "${STAGES_NAMES[$i]} : $P% failure rate${NC}"
done

echo "=========================================="
echo "=========================================="
echo "=========================================="
