#!/usr/bin/env bash
set -e

# Default values
DEFAULT_REPO_URL="https://github.com/sunnypilot"
START_AT_BOOT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-at-boot)
            START_AT_BOOT=true
            shift
            ;;
        --token)
            GITHUB_TOKEN="$2"
            shift 2
            ;;
        --repo)
            REPO_URL="$2"
            shift 2
            ;;
        *)
            if [ -z "$GITHUB_TOKEN" ]; then
                GITHUB_TOKEN="$1"
            elif [ -z "$REPO_URL" ]; then
                REPO_URL="$1"
            fi
            shift
            ;;
    esac
done

# Check required arguments
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Usage: $0 [--start-at-boot] [--token <github_token>] [--repo <repository_url>]"
    echo "Required argument: github_token"
    echo "Optional arguments:"
    echo "  --start-at-boot    Enable auto-start at boot (default: false)"
    echo "  --repo            Repository URL (default: ${DEFAULT_REPO_URL})"
    exit 1
fi

# Set repository URL if not provided
REPO_URL="${REPO_URL:-$DEFAULT_REPO_URL}"

# Constants
RUNNER_USER="github-runner"
USER_GROUPS="comma,gpu,gpio,sudo"
BASE_DIR="/data/github"
RUNNER_DIR="${BASE_DIR}/runner"
BUILDS_DIR="${BASE_DIR}/builds"
LOGS_DIR="${BASE_DIR}/logs"
CACHE_DIR="${BASE_DIR}/cache"
OPENPILOT_DIR="${BASE_DIR}/openpilot"

create_directories() {
    sudo mkdir -p "$RUNNER_DIR" "$BUILDS_DIR" "$LOGS_DIR" "$CACHE_DIR" "$OPENPILOT_DIR"
    mkdir -p "/data/openpilot"
    sudo chown -R comma:comma "/data/openpilot"
}

download_and_setup_runner() {
    cd "$RUNNER_DIR"
    curl -o actions-runner-linux-arm64-2.321.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.321.0/actions-runner-linux-arm64-2.321.0.tar.gz
    tar xzf ./actions-runner-linux-arm64-2.321.0.tar.gz
    rm ./actions-runner-linux-arm64-2.321.0.tar.gz
    chmod +x ./config.sh
}

setup_runner_user() {
    sudo useradd --comment 'GitHub Runner' --create-home --home-dir ${BASE_DIR} ${RUNNER_USER} --shell /bin/bash -G ${USER_GROUPS} || sudo usermod -aG ${USER_GROUPS} ${RUNNER_USER}
    export BASE_DIR
    sudo -u ${RUNNER_USER} bash -c "truncate -s 0 '${BASE_DIR}/.bash_logout'"
}

create_sudoers_entry() {
    sudo grep -qxF "${RUNNER_USER} ALL=(ALL) NOPASSWD: ALL" /etc/sudoers || echo "${RUNNER_USER} ALL=(ALL) NOPASSWD: ALL" | sudo tee -a /etc/sudoers
}

configure_runner() {
    cd "$RUNNER_DIR"
    sudo -u ${RUNNER_USER} ./config.sh --url "$REPO_URL" --token "$GITHUB_TOKEN" --name $(hostname) --runnergroup "tici-tizi" --labels "tici" --work "$BUILDS_DIR" --unattended
}

set_directory_permissions() {
    sudo chown -R ${RUNNER_USER}:comma "$BASE_DIR"
    sudo chmod g+rwx "$BASE_DIR"
    sudo chmod g+s "$BASE_DIR"
}

modify_service_template() {
    cat <<EOL > "$RUNNER_DIR/bin/actions.runner.service.template"
[Unit]
Description={{Description}}
After=network-online.target nss-lookup.target time-sync.target
Wants=network-online.target nss-lookup.target time-sync.target
StartLimitInterval=5
StartLimitBurst=10

[Service]
Type=simple
User=root
ExecStart=/usr/bin/unshare -m -- /bin/bash -c 'mount --bind ${OPENPILOT_DIR} /data/openpilot && setpriv --reuid={{User}} --regid={{User}} --init-groups env HOME=${BASE_DIR} USER={{User}} LOGNAME={{User}} MAIL=/var/mail/{{User}} {{RunnerRoot}}/runsvc.sh'
WorkingDirectory={{RunnerRoot}}
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min
Restart=always
RestartSec=120

[Install]
WantedBy=multi-user.target
EOL
}

# Make filesystem writable
sudo mount -o remount,rw /

# Ensure filesystem is remounted as read-only on script exit
trap "sudo mount -o remount,ro /" EXIT

# Execute installation steps
setup_runner_user
create_sudoers_entry
create_directories
download_and_setup_runner
modify_service_template
configure_runner
set_directory_permissions

# Install and start service using built-in installer
cd "$RUNNER_DIR"
sudo ./svc.sh install $RUNNER_USER

# Handle auto-start configuration
if [ "$START_AT_BOOT" = false ]; then
    sudo systemctl disable actions.runner.sunnypilot.$(uname -n)
fi

sudo ./svc.sh start