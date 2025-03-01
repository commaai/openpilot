#!/usr/bin/env bash
set -e

# Default values
DEFAULT_REPO_URL="https://github.com/sunnypilot"
START_AT_BOOT=false
RESTORE_MODE=false

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
        --restore)
            RESTORE_MODE=true
            shift
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

# Determine BASE_DIR based on mount point
if mountpoint -q /data/media; then
    BASE_DIR="/data/media/0/github"
else
    BASE_DIR="/data/github"
fi

# Constants
RUNNER_USER="github-runner"
USER_GROUPS="comma,gpu,gpio,sudo"
RUNNER_DIR="${BASE_DIR}/runner"
BUILDS_DIR="${BASE_DIR}/builds"
LOGS_DIR="${BASE_DIR}/logs"
CACHE_DIR="${BASE_DIR}/cache"
OPENPILOT_DIR="${BASE_DIR}/openpilot"

# Basic utility functions (no dependencies)
remount_rw() {
    sudo mount -o remount,rw /
}

remount_ro() {
    sync || true  # Try to sync but continue even if it fails
    sudo mount -o remount,ro /  # Always try to remount as read-only
}

# Always ensure we try to remount as read-only on exit
trap remount_ro EXIT

setup_runner_user() {
    sudo useradd --comment 'GitHub Runner' --create-home --home-dir ${BASE_DIR} ${RUNNER_USER} --shell /bin/bash -G ${USER_GROUPS} || sudo usermod -aG ${USER_GROUPS} ${RUNNER_USER}
}

create_sudoers_entry() {
    sudo grep -qxF "${RUNNER_USER} ALL=(ALL) NOPASSWD: ALL" /etc/sudoers || echo "${RUNNER_USER} ALL=(ALL) NOPASSWD: ALL" | sudo tee -a /etc/sudoers
}

set_directory_permissions() {
    sudo chown -R ${RUNNER_USER}:comma "$BASE_DIR"
    sudo chmod -R g+rwx "$BASE_DIR"
    sudo find "$BASE_DIR" -type d -exec chmod g+s {} +
}

setup_directories() {
    echo "Creating necessary directories..."
    sudo mkdir -p "$RUNNER_DIR" "$BUILDS_DIR" "$LOGS_DIR" "$CACHE_DIR" "$OPENPILOT_DIR"
    mkdir -p "/data/openpilot"
    sudo chown -R comma:comma "/data/openpilot"
    sync
}

wipe_bash_logout() {
  export BASE_DIR
  sudo -u ${RUNNER_USER} bash -c "touch ${BASE_DIR}/.bash_logout"
  sudo -u ${RUNNER_USER} bash -c "truncate -s 0 '${BASE_DIR}/.bash_logout'"
}

# System configuration functions (depends on basic utility functions)
setup_system_configs() {
    echo "Setting up system configurations..."
    remount_rw
    setup_runner_user
    create_sudoers_entry
    remount_ro
    set_directory_permissions
    wipe_bash_logout
}

# Runner setup functions
install_runner() {
    echo "Downloading and setting up runner..."
    cd "$RUNNER_DIR"
    curl -o actions-runner-linux-arm64-2.322.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-arm64-2.322.0.tar.gz
    sudo -u ${RUNNER_USER} tar -xzf ./actions-runner-linux-arm64-2.322.0.tar.gz
    sudo rm ./actions-runner-linux-arm64-2.322.0.tar.gz
    sudo chmod +x ./config.sh
}

configure_runner() {
    remount_rw
    echo "Configuring runner..."
    cd "$RUNNER_DIR"
    sudo -u ${RUNNER_USER} ./config.sh --url "$REPO_URL" --token "$GITHUB_TOKEN" --name $(hostname) --runnergroup "tici-tizi" --labels "tici" --work "$BUILDS_DIR" --unattended
    remount_ro
}

create_service_template() {
    echo "Creating service template..."
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

install_service() {
    remount_rw
    echo "Installing systemd service..."
    cd "$RUNNER_DIR"
    sudo ./svc.sh install $RUNNER_USER

    if [ "$START_AT_BOOT" = false ]; then
        local service_name
        if [ -f "${RUNNER_DIR}/.service" ]; then
            service_name=$(cat "${RUNNER_DIR}/.service")
        else
            service_name="actions.runner.sunnypilot.$(uname -n)"
        fi
        sudo systemctl disable "${service_name}"
    fi
    remount_ro
}

check_restore_prerequisites() {
    local needs_restore=false
    local can_restore=false
    local service_name=""

    # Check if base runner directory exists
    if [ ! -d "${RUNNER_DIR}" ]; then
        echo "ERROR: Runner directory ${RUNNER_DIR} does not exist"
        echo "This directory is required for restore operations"
        exit 1
    fi

    # First check if we have the required files for restoration
    if [ -f "${RUNNER_DIR}/.credentials" ] && [ -f "${RUNNER_DIR}/.service" ]; then
        can_restore=true
        service_name=$(cat "${RUNNER_DIR}/.service")
        echo "Found required runner configuration files"
    else
        echo "Missing required runner configuration files"
        echo "Required: .credentials and .service files in ${RUNNER_DIR}"
        exit 1
    fi

    # Then check if restoration is needed (if either service or user is missing)
    if ! systemctl list-unit-files "${service_name}" &>/dev/null; then
        echo "Service ${service_name} not found in systemd"
        needs_restore=true
    fi

    if ! id "${RUNNER_USER}" &>/dev/null; then
        echo "User ${RUNNER_USER} does not exist"
        needs_restore=true
    fi

    # Only proceed if we can restore AND need to restore
    if [ "$can_restore" = true ] && [ "$needs_restore" = true ]; then
        echo "Restoration is needed and possible"
        return 0
    else
        if [ "$needs_restore" = false ]; then
            echo "System is already properly configured (user and service exist)"
        fi
        exit 0
    fi
}

perform_restore() {
    echo "Starting runner restoration..."
    setup_directories
    setup_system_configs
    install_service
    echo "Runner restoration completed successfully"
}

perform_install() {
    echo "Starting fresh installation..."
    setup_directories
    setup_system_configs
    install_runner
    set_directory_permissions
    create_service_template
    configure_runner
    install_service
    echo "Installation completed successfully"
}

main() {
    if [ "$RESTORE_MODE" = true ]; then
        echo "Running in restore mode - will only restore system configurations..."
        check_restore_prerequisites
        perform_restore
    else
        # Check required arguments for normal installation
        if [ -z "$GITHUB_TOKEN" ]; then
            echo "Usage: $0 [--start-at-boot] [--token <github_token>] [--repo <repository_url>] [--restore]"
            echo "Required argument (except for --restore): github_token"
            echo "Optional arguments:"
            echo "  --start-at-boot    Enable auto-start at boot (default: false)"
            echo "  --repo            Repository URL (default: ${DEFAULT_REPO_URL})"
            echo "  --restore         Restore existing runner configuration"
            exit 1
        fi

        # Set repository URL if not provided
        REPO_URL="${REPO_URL:-$DEFAULT_REPO_URL}"
        perform_install
    fi

    echo "Starting runner service..."
    cd "$RUNNER_DIR"
    sudo ./svc.sh start
}

main
