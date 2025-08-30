#!/usr/bin/env bash
set -e

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
UNDERLINE='\033[4m'
BOLD='\033[1m'
NC='\033[0m'

BRANCH="master"
RUNS="20"

COOKIE_JAR=/tmp/cookies
CRUMB=$(curl -s --cookie-jar $COOKIE_JAR 'https://jenkins.comma.life/crumbIssuer/api/xml?xpath=concat(//crumbRequestField,":",//crumb)')

FIRST_LOOP=1

function loop() {
  JENKINS_BRANCH="__jenkins_loop_${BRANCH}_$(date +%s)"
  API_ROUTE="https://jenkins.comma.life/job/openpilot/job/$JENKINS_BRANCH"

  for run in $(seq 1 $((RUNS / 2))); do

    N=2

    if [[ $FIRST_LOOP ]]; then
      TEMP_DIR=$(mktemp -d)
      GIT_LFS_SKIP_SMUDGE=1 git clone --quiet -b $BRANCH --depth=1 --no-tags git@github.com:commaai/openpilot $TEMP_DIR
      git -C $TEMP_DIR checkout --quiet -b $JENKINS_BRANCH
      echo "TESTING: $(date)" >> $TEMP_DIR/testing_jenkins
      git -C $TEMP_DIR add testing_jenkins
      git -C $TEMP_DIR commit --quiet -m "testing"
      git -C $TEMP_DIR push --quiet -f origin $JENKINS_BRANCH
      rm -rf $TEMP_DIR
      FIRST_BUILD=1
      echo ''
      echo 'waiting on Jenkins...'
      echo ''
      sleep 90
      FIRST_LOOP=""
    fi

    FIRST_BUILD=$(curl -s $API_ROUTE/api/json | jq .nextBuildNumber)
    LAST_BUILD=$((FIRST_BUILD+N-1))
    TEST_BUILDS=( $(seq $FIRST_BUILD $LAST_BUILD) )

    # Start N new builds
    for i in ${TEST_BUILDS[@]};
    do
      echo "Starting build $i"
      curl -s --output /dev/null --cookie $COOKIE_JAR -H "$CRUMB" -X POST $API_ROUTE/build?delay=0sec
      sleep 5
    done
    echo ""

    # Wait for all builds to end
    while true; do
      sleep 30

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
      echo "See live results: ${API_ROUTE}/buildTimeTrend"
      echo ""

      if [[ $count -ge $N ]]; then
        break
      fi
    done

  done
}

function usage() {
  echo ""
  echo "Run the Jenkins tests multiple times on a specific branch"
  echo ""
  echo -e "${BOLD}${UNDERLINE}Options:${NC}"
  echo -e "  ${BOLD}-n, --n${NC}"
  echo -e "          Specify how many runs to do (default to ${BOLD}20${NC})"
  echo -e "  ${BOLD}-b, --branch${NC}"
  echo -e "          Specify which branch to run the tests against (default to ${BOLD}master${NC})"
  echo ""
}

function _looper() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 0
  fi

  # parse Options
  while [[ $# -gt 0 ]]; do
    case $1 in
      -n | --n ) shift 1; RUNS="$1"; shift 1 ;;
      -b | --b | --branch | -branch ) shift 1; BRANCH="$1"; shift 1 ;;
      * ) usage; exit 0 ;;
    esac
  done

  echo ""
  echo -e "You are about to start $RUNS Jenkins builds against the $BRANCH branch."
  echo -e "If you expect this to run overnight, ${UNDERLINE}${BOLD}unplug the cold reboot power switch${NC} from the testing closet before."
  echo ""
  read -p "Press (y/Y) to confirm: " choice
  if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    loop
  fi

}

_looper $@
