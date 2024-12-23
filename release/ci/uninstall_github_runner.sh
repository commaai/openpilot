#!/usr/bin/env bash
# Determine BASE_DIR based on mount point
if mountpoint -q /data/media; then
    GITHUB_BASE_DIR="/data/media/0/github"
else
    GITHUB_BASE_DIR="/data/github"
fi

# Define directories and user
BIN_DIR="$GITHUB_BASE_DIR/bin"
BUILDS_DIR="$GITHUB_BASE_DIR/builds"
OPENPILOT_DIR="$GITHUB_BASE_DIR/openpilot"
LOGS_DIR="$GITHUB_BASE_DIR/logs"
CACHE_DIR="$GITHUB_BASE_DIR/cache"
RUNNER_USERNAME="github-runner"
# Define the systemd service name
SERVICE_NAME="github-runner"
USER_GROUPS="comma,gpu,gpio,sudo"

# Function to stop and disable the systemd service
stop_and_uninstall_service() {
    cd $GITHUB_BASE_DIR/runner
    sudo ./svc.sh stop
    sudo ./svc.sh uninstall
}

# Function to remove the systemd service file
remove_runner() {
    cd $GITHUB_BASE_DIR/runner
    sudo rm .runner
    sudo su -c './config.sh remove' github-runner
}

# Function to delete the Github Runner directories
delete_directories() {
    sudo rm -rf "$BIN_DIR/github-runner"
    sudo rm -rf "$GITHUB_BASE_DIR" "$BIN_DIR" "$BUILDS_DIR" "$LOGS_DIR" "$CACHE_DIR" "$OPENPILOT_DIR"
}

# Function to remove the Github Runner user
delete_user() {
    for group in ${USER_GROUPS//,/ }
    do
       sudo gpasswd -d ${RUNNER_USERNAME} ${group}
    done
    sudo userdel -r ${RUNNER_USERNAME}
}

# Function to remove sudoers entry
remove_sudoers_entry() {
    sudo sed -i.bak "/${RUNNER_USERNAME} ALL=(ALL) NOPASSWD: ALL/d" /etc/sudoers
}

# Make filesystem writable
sudo mount -o remount rw /

# Ensure filesystem is remounted as read-only on script exit
trap "sudo mount -o remount ro /" EXIT

# Call functions
stop_and_uninstall_service
remove_runner
delete_directories
delete_user
remove_sudoers_entry
# End of uninstall script
