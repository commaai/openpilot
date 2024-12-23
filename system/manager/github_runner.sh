#!/usr/bin/env bash

# Define the service name
SERVICE_NAME="actions.runner.sunnypilot.$(uname -n)"

# Function to control the service
control_service() {
    local action=$1  # Store the function argument in a local variable
    sudo systemctl $action ${SERVICE_NAME}
}

service_exists_and_is_loaded() {
    sudo systemctl status ${SERVICE_NAME} &>/dev/null
    if [[ $? -ne 4 ]]; then
        return 0  # Service is known to systemd (i.e., loaded)
    else
        return 1  # Service is unknown to systemd (i.e., not loaded)
    fi
}

# Check for required argument
if [[ -z $1 ]] || { [[ $1 != "start" ]] && [[ $1 != "stop" ]]; }; then
    echo "Usage: $0 {start|stop}"
    exit 1
fi

# Store the script argument in a descriptive variable
ACTION=$1

# Trap EXIT signal (Ctrl+C) and stop the service
trap 'control_service stop ; exit' SIGINT SIGKILL EXIT

# Enter the main loop
while true; do
    # Check if the service is actually present on the system
    if service_exists_and_is_loaded; then
        control_service $ACTION  # Call the function with the specified action
    fi
    sleep 1  # Pause before the next iteration
done