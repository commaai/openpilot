#!/bin/sh
install_loc="$HOME/.local/bin"
docker build -t qemu-hexagon-static:latest - <<'EOF'
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends qemu-user-static ca-certificates && rm -rf /var/lib/apt/lists/*
EOF

mkdir -p "$install_loc"
tee "$install_loc/qemu-hexagon-static" >/dev/null <<'EOF'
#!/bin/sh
set -eu
exec docker run --rm -i \
  -v /var/folders:/var/folders -v "$HOME":"$HOME" \
  qemu-hexagon-static:latest qemu-hexagon-static "$@"
EOF
chmod +x "$install_loc/qemu-hexagon-static"
