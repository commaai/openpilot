set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR"

echo "Setting up beampilot dependencies and environment..."

if [ -f "$DIR/tools/op.sh" ]; then
  # this installs everything pretty much
  # and runs everything too, like uv lock, etc
  "$DIR/tools/op.sh" setup
else
  # tools is installed in stock OP too, not only beampilot
  echo "tools/op.sh not found (included in repo)"
  exit 1
fi

echo "beampilot setup complete"
