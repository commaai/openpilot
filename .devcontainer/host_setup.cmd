:: pull base image
IF NOT DEFINED USE_LOCAL_IMAGE ^
echo "Updating openpilot_base image if needed..." && ^
docker pull ghcr.io/commaai/openpilot-base:latest

:: setup .host dir
mkdir .devcontainer\.host

:: setup host env file
echo "" > .devcontainer\.host\.env