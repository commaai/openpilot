DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source "$DIR/config_beampilot.sh"

echo "Starting..."
exec ./launch_chffrplus.sh