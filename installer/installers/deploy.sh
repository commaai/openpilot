#!/usr/bin/env bash
set -e

ENVIRONMENT="${1}"
if [ "${ENVIRONMENT}" != "staging" -a "${ENVIRONMENT}" != "prod" ]; then
	echo "usage: $0 <env>" >&2
	echo "  <env> = staging or prod" >&2
	exit 1
fi

SUFFIX=""
if [ "${ENVIRONMENT}" != "prod" ]; then
  SUFFIX="_test"
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ -z $(az account show 2>/dev/null) ]]; then
  echo "$(date --rfc-3339=s) LOGIN: azure"
  az login
fi

FILES=(
installer_openpilot
installer_dashcam
)
for FILE in ${FILES[@]}; do
  KEY="${FILE}${SUFFIX}"
  echo "$(date --rfc-3339=s) PUSHING: ${FILE} -> ${KEY}"
  az storage blob upload \
    --account-name commadist \
    --container-name neosupdate \
    --name "${KEY}" \
    --file "${FILE}"
done
